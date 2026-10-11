"""Fixed paired background-crop retraining with a frozen teacher and old positive gates."""
import argparse
import hashlib
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from scripts.blue_objects_round2_trial_data import inputs
from scripts.blue_object_round2_data import ROOT, QUEUE, REVIEW
from scripts.train_indoor5_trial import calculate as prior_calculate
from scripts.balloon_all_correct_protection import all_correct_loss

PROTOCOL = Path('configs/blue-objects-round2-trial-20261011.json')
GROUPS = ('', 'blue_', 'clutter_', 'real_', 'new_', 'runtime_', 'photo_', 'batch_', 'indoor_', 'object_')


def run_path(mode, seed):
    p = read_json(PROTOCOL)
    if mode not in p['arms'] or seed not in p['seeds']:
        raise ValueError('Undeclared arm or seed')
    return Path(f'runs/blue-objects-round2-trial-{mode}-seed{seed}-20261011')


def calculate(arrays):
    base = dict(arrays)
    for key in ('before', 'after', 'labels', 'ordinary'):
        base[key] = arrays[key][:1922]
    report = prior_calculate(base)
    values = arrays['object_after']
    if values.shape != (12, 3) or not np.isfinite(values).all():
        raise ValueError('Invalid new background probes')
    report['object_background_correct'] = int((values.argmax(1) == 0).sum())
    report['training_gate_pass'] = report['training_gate_pass'] and report['object_background_correct'] == 12
    return report


def train(mode, seed):
    p, out = read_json(PROTOCOL), run_path(mode, seed)
    if out.exists():
        raise FileExistsError(out)
    if sha256(p['parent']) != p['parent_sha256']:
        raise ValueError('Frozen teacher changed')
    rows, labels, ordinary, tensors = inputs(mode)
    steps = int(np.ceil(len(rows) / p['batch_size'])) * p['epochs']
    if len(rows) != 1946 or len(tensors) != 10 or steps != p['decay_steps'] or steps != 48800:
        raise ValueError('Unexpected curriculum or schedule')
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    keras.utils.set_random_seed(seed)
    model = keras.models.load_model(p['parent'], compile=False)
    if model.count_params() != p['parameters'] or p['parameters'] != 7763:
        raise ValueError('Architecture changed')
    arrays = {k + 'before': model(x, training=False).numpy() for k, x in zip(GROUPS, tensors, strict=True)}
    arrays.update(labels=labels, ordinary=ordinary, indoor_labels=labels[1786:1854])
    targets = np.column_stack((np.eye(3)[labels], arrays['before'], ordinary)).astype(np.float32)
    inherited = Path('runs/indoor5-trial-admit-seed47-20261011/provenance.json')
    sources = read_json(inherited)['sources_sha256'].copy()
    for path in (PROTOCOL, inherited, REVIEW, ROOT / 'acquisition.json', ROOT / 'license-verification.json',
                 QUEUE / 'review.json', 'scripts/blue_objects_round2_trial_data.py',
                 'scripts/blue_object_round2_data.py', 'scripts/train_blue_objects_round2_trial.py',
                 'research/balloon-red-blue-20261008/blue-objects-round2-admission-20261011.json'):
        sources[str(path)] = sha256(path)
    if any(sha256(path) != digest for path, digest in sources.items()):
        raise ValueError('Inherited input provenance changed')
    provenance = dict(mode=mode, seed=seed, settings=p, parameters=model.count_params(),
        sample_identity=[{k: r[k] for k in ('source_image', 'sha256', 'label', 'preserve', 'view')} for r in rows],
        sources_sha256=sources, tensor_sha256=[hashlib.sha256(x.tobytes()).hexdigest() for x in tensors],
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
        model.save(out / 'student.keras')
        np.savez_compressed(out / 'predictions.npz', **arrays)
        report.update(parameters=model.count_params(), model_sha256=sha256(out / 'student.keras'),
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
