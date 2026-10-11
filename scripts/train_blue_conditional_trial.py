"""Paired blue correction; all parent weights and parent red probabilities remain frozen."""
import argparse
import hashlib
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from dtr.runtime import softmax
from scripts.blue_batch_trial_data import inputs
from scripts.blue_balloon_batch_data import ROOT, QUEUE, REVIEW
from scripts.train_blue_batch_trial import calculate as prior_calculate
from scripts.blue_conditional_adapter import build
from scripts.balloon_all_correct_protection import all_correct_loss

PROTOCOL = Path('configs/blue-conditional-trial-20261011.json')
GROUPS = ('', 'blue_', 'clutter_', 'real_', 'new_', 'runtime_', 'photo_', 'batch_')


def run_path(mode, seed):
    p = read_json(PROTOCOL)
    if mode not in p['arms'] or seed not in p['seeds']:
        raise ValueError('Undeclared arm or seed')
    return Path(f'runs/blue-conditional-trial-{mode}-seed{seed}-20261011')


def frozen_weights(model, names):
    return {name: [dict(shape=list(w.shape), dtype=str(w.dtype),
        sha256=hashlib.sha256(w.tobytes()).hexdigest()) for w in model.get_layer(name).get_weights()]
        for name in names}


def calculate(arrays):
    report = prior_calculate(arrays)
    delta = 0.
    for prefix in GROUPS:
        before, after = softmax(arrays[prefix+'before']), softmax(arrays[prefix+'after'])
        if before.shape != after.shape or not np.isfinite(after).all():
            raise ValueError('Invalid red-probability check')
        delta = max(delta, float(np.max(np.abs(before[:,1]-after[:,1]))))
        if delta > 1e-6 or not np.array_equal(before[:,1] >= .8, after[:,1] >= .8):
            raise ValueError('Frozen red probability or acceptance changed')
    report['max_red_probability_delta'] = delta
    report['red_acceptance_unchanged'] = True
    return report


def train(mode, seed):
    p, out = read_json(PROTOCOL), run_path(mode, seed)
    if out.exists():
        raise FileExistsError(out)
    if sha256(p['parent']) != p['parent_sha256']:
        raise ValueError('Frozen teacher changed')
    rows, labels, ordinary, tensors = inputs(mode)
    steps = int(np.ceil(len(rows) / p['batch_size'])) * p['epochs']
    if len(rows) != 1786 or len(tensors) != 8 or steps != p['decay_steps'] or steps != 44800:
        raise ValueError('Unexpected curriculum or schedule')
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    keras.utils.set_random_seed(seed)
    model = keras.models.load_model(p['parent'], compile=False)
    if model.count_params() != 7763:
        raise ValueError('Architecture changed')
    arrays = {k + 'before': model(x, training=False).numpy() for k, x in zip(GROUPS, tensors, strict=True)}
    parent_names = [layer.name for layer in model.layers]
    frozen_before = frozen_weights(model, parent_names)
    model = build(model)
    if model.count_params() != p['parameters'] or p['parameters'] != 9844:
        raise ValueError('Conditional architecture changed')
    if sum(int(np.prod(w.shape)) for w in model.trainable_weights) != 2081:
        raise ValueError('Parent must remain frozen')
    for prefix, x in zip(GROUPS, tensors, strict=True):
        expected, actual = softmax(arrays[prefix+'before']), softmax(model(x, training=False).numpy())
        if (not np.allclose(expected, actual, rtol=0, atol=1e-6)
                or not np.array_equal(expected.argmax(1), actual.argmax(1))
                or not np.array_equal(expected.max(1)>=.8, actual.max(1)>=.8)):
            raise ValueError('Initial decisions changed')
    arrays.update(labels=labels, ordinary=ordinary)
    targets = np.column_stack((np.eye(3)[labels], arrays['before'], ordinary)).astype(np.float32)
    inherited = Path('runs/dense-context-trial-control-seed47-20261011/provenance.json')
    sources = read_json(inherited)['sources_sha256'].copy()
    for path in (PROTOCOL, inherited, REVIEW, ROOT / 'acquisition.json', ROOT / 'license-verification.json',
                 QUEUE / 'review.json', 'scripts/blue_batch_trial_data.py', 'scripts/blue_balloon_batch_data.py',
                 'scripts/train_dense_context_trial.py', 'scripts/train_blue_conditional_trial.py',
                 'scripts/train_blue_batch_trial.py', 'scripts/blue_conditional_adapter.py',
                 'research/balloon-red-blue-20261008/blue-conditional-initialization-20261011.json',
                 'research/balloon-red-blue-20261008/blue-balloon-batch2-admission-20261011.json'):
        sources[str(path)] = sha256(path)
    if any(sha256(path) != digest for path, digest in sources.items()):
        raise ValueError('Inherited input provenance changed')
    provenance = dict(mode=mode, seed=seed, settings=p, parameters=model.count_params(),
        sample_identity=[{k: r[k] for k in ('source_image', 'sha256', 'label', 'preserve', 'view')} for r in rows],
        sources_sha256=sources, tensor_sha256=[hashlib.sha256(x.tobytes()).hexdigest() for x in tensors],
        frozen_parent_layers=frozen_before, trainable_parameters=2081,
        optimizer_reset=True, final_epoch_only=True, test_evaluated=False, deployment_approved=False)
    out.mkdir(parents=True)
    write_json(out / 'provenance.json', provenance)
    write_json(out / 'status.json', dict(state='running'))
    try:
        keras.utils.set_random_seed(seed)
        dataset = tf.data.Dataset.from_tensor_slices((tensors[0], targets,
            np.where(ordinary, 1., p['hard_weight']).astype(np.float32)))
        dataset = dataset.shuffle(len(rows), seed=seed).batch(p['batch_size'])
        options = tf.data.Options()
        options.threading.private_threadpool_size = 2
        dataset = dataset.with_options(options).prefetch(1)
        schedule = keras.optimizers.schedules.CosineDecay(p['learning_rate'], steps, alpha=p['alpha'])
        model.compile(optimizer=keras.optimizers.Adam(schedule),
            loss=all_correct_loss(p['protection_weight'], p['protection_probability_cap'], p['parent_acceptance_threshold']))

        class Progress(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if not logs or not all(np.isfinite(v) for v in logs.values()):
                    raise ValueError('Nonfinite training')
                write_json(out / 'status.json', dict(state='running', epoch=epoch + 1, metrics=logs))

        model.fit(dataset, epochs=p['epochs'], verbose=2,
                  callbacks=[Progress(), keras.callbacks.CSVLogger(str(out / 'epochs.csv'))])
        arrays.update({k + 'after': model(x, training=False).numpy() for k, x in zip(GROUPS, tensors, strict=True)})
        report = calculate(arrays)
        if frozen_weights(model, parent_names) != frozen_before:
            raise ValueError('Frozen parent weights changed during fitting')
        model.save(out / 'student.keras')
        np.savez_compressed(out / 'predictions.npz', **arrays)
        report.update(parameters=model.count_params(), trainable_parameters=2081, frozen_parent_unchanged=True, model_sha256=sha256(out / 'student.keras'),
            predictions_sha256=sha256(out / 'predictions.npz'), provenance_sha256=sha256(out / 'provenance.json'),
            optimizer_steps=int(model.optimizer.iterations.numpy()),
            scope='TRAIN crop feasibility only; not real-scene accuracy or Pi readiness',
            test_evaluated=False, deployment_approved=False)
        if report['optimizer_steps'] != steps or any(sha256(path) != h for path, h in sources.items()):
            raise ValueError('Steps or source identities changed')
        write_json(out / 'report.json', report)
        write_json(out / 'status.json', dict(state='complete'))
        print(dict(mode=mode, seed=seed, report=report), flush=True)
    except BaseException as exc:
        write_json(out / 'status.json', dict(state='failed', error=str(exc)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', required=True, choices=('control', 'admit'))
    parser.add_argument('--seed', required=True, type=int, choices=(47, 48))
    args = parser.parse_args()
    train(args.mode, args.seed)
