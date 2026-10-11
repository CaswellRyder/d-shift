"""Paired new-blue TRAIN expansion from a fixed clutter-trained warm start."""
import argparse
import hashlib
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from dtr.runtime import softmax
from scripts.openimages_blue_expansion import data, REVIEW, PREVIEW
from scripts.grid_clutter_data import load_crop
from scripts.balloon_all_correct_protection import all_correct_loss
from scripts.balloon_positive_protection import retention, feasible
from scripts.train_balloon_reviewed_clutter import CLASSES

PROTOCOL = Path('configs/balloon-blue-expansion2-trial-20261011.json')


def training_gate(report, protocol):
    return feasible(report, protocol) and report['new_blue_accepted'] >= protocol['minimum_new_blue_accepted']


def train(mode, seed, output):
    protocol, out = read_json(PROTOCOL), Path(output)
    if out.exists():
        raise FileExistsError(out)
    if mode not in protocol['arms'] or seed not in protocol['seeds']:
        raise ValueError('Undeclared trial arm or seed')
    for name in ('parent', 'teacher'):
        if sha256(protocol[name]) != protocol[name + '_sha256']:
            raise ValueError('Fixed checkpoint changed')
    rows, extra, clutter = data(mode)
    ordinary = np.array([r['preserve'] for r in rows], bool)
    if len(rows) != 1577 or ordinary.sum() != 1487 or len(extra) != 23:
        raise ValueError('Matched dataset counts changed')
    x = np.stack([load_crop(r)[0] for r in rows])
    labels = np.array([CLASSES.index(r['label']) for r in rows])
    probes = np.stack([load_crop(dict(r, view=v))[0] for v in ('original', 'blue_surface') for r in clutter])
    blue = np.stack([load_crop(r)[0] for r in extra])
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    keras.utils.set_random_seed(seed)
    teacher = keras.models.load_model(protocol['teacher'], compile=False)
    model = keras.models.load_model(protocol['parent'], compile=False)
    if model.count_params() != 7763 or model.output_shape != (None, 3):
        raise ValueError('Architecture changed')
    before = teacher(x, training=False).numpy()
    start = model(x, training=False).numpy()
    probe_before, blue_before = model(probes, training=False).numpy(), model(blue, training=False).numpy()
    targets = np.column_stack((np.eye(3)[labels], before, ordinary)).astype(np.float32)
    sources = [str(PROTOCOL), str(REVIEW), str(PREVIEW / 'preview.json'),
        'scripts/openimages_blue_expansion.py', 'scripts/grid_clutter_data.py',
        'scripts/balloon_all_correct_protection.py', 'scripts/balloon_positive_protection.py',
        'scripts/train_balloon_reviewed_clutter.py', 'scripts/train_balloon_fragments.py', __file__]
    identity = [{k: r[k] for k in ('source_image', 'sha256', 'label', 'preserve', 'view')} for r in rows]
    provenance = dict(mode=mode, seed=seed, settings=protocol, parameters=model.count_params(),
        sources_sha256={p: sha256(p) for p in sources}, sample_identity=identity,
        input_tensor_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
        probe_tensor_sha256=hashlib.sha256(probes.tobytes()).hexdigest(),
        blue_tensor_sha256=hashlib.sha256(blue.tobytes()).hexdigest(),
        optimizer_reset=True, final_epoch_only=True, test_evaluated=False, deployment_approved=False)
    out.mkdir(parents=True)
    write_json(out / 'provenance.json', provenance)
    write_json(out / 'status.json', dict(state='running'))
    try:
        weights = np.where(ordinary, 1., protocol['hard_weight']).astype(np.float32)
        dataset = tf.data.Dataset.from_tensor_slices((x, targets, weights)).shuffle(len(x), seed=seed).batch(protocol['batch_size'])
        options = tf.data.Options()
        options.threading.private_threadpool_size = 2
        dataset = dataset.with_options(options).prefetch(1)
        model.compile(optimizer=keras.optimizers.Adam(protocol['learning_rate']),
            loss=all_correct_loss(protocol['protection_weight'], protocol['protection_probability_cap'],
                                  protocol['parent_acceptance_threshold']))

        class Progress(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if not logs or not all(np.isfinite(v) for v in logs.values()):
                    raise ValueError('Nonfinite metrics')
                write_json(out / 'status.json', dict(state='running', epoch=epoch + 1, metrics=logs))

        model.fit(dataset, epochs=protocol['epochs'], verbose=2,
            callbacks=[Progress(), keras.callbacks.CSVLogger(str(out / 'epochs.csv'))])
        after = model(x, training=False).numpy()
        probe_after, blue_after = model(probes, training=False).numpy(), model(blue, training=False).numpy()
        if any(not np.isfinite(a).all() for a in (before, start, after, probe_after, blue_after)):
            raise ValueError('Nonfinite checkpoint predictions')
        model.save(out / 'student.keras')
        np.savez_compressed(out / 'predictions.npz', before=before, start=start, after=after,
            labels=labels, ordinary=ordinary, probe_before=probe_before, probe_after=probe_after,
            blue_before=blue_before, blue_after=blue_after)
        report = dict(retention=retention(before, after, labels, ordinary),
            warm_start_retention=retention(start, after, labels, ordinary),
            reviewed_background_correct=int((probe_after.argmax(1) == 0).sum()),
            new_blue_correct=int((blue_after.argmax(1) == 2).sum()),
            new_blue_accepted=int(((blue_after.argmax(1) == 2) & (softmax(blue_after)[:, 2] >= .8)).sum()),
            initial_blue_accepted=int(((blue_before.argmax(1) == 2) & (softmax(blue_before)[:, 2] >= .8)).sum()),
            model_sha256=sha256(out / 'student.keras'), predictions_sha256=sha256(out / 'predictions.npz'),
            provenance_sha256=sha256(out / 'provenance.json'),
            scope='TRAIN feasibility only; public blue photo expansion is not competition qualification',
            test_evaluated=False, deployment_approved=False)
        report['training_gate_pass'] = training_gate(report, protocol)
        if any(sha256(p) != h for p, h in provenance['sources_sha256'].items()):
            raise ValueError('Sources changed while training')
        write_json(out / 'report.json', report)
        write_json(out / 'status.json', dict(state='complete'))
        print(dict(mode=mode, seed=seed, report=report))
    except BaseException as exc:
        write_json(out / 'status.json', dict(state='failed', error=str(exc)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('control', 'admit'), required=True)
    parser.add_argument('--seed', type=int, choices=(47, 48), required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    train(args.mode, args.seed, args.output)
