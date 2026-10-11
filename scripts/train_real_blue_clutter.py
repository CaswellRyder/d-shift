"""Matched real-blue background trial protecting previously learned positives."""
import argparse
import hashlib
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from dtr.runtime import softmax
from scripts.blue_false_alarm_data import data, REVIEW, QUEUE
from scripts.grid_clutter_data import load_crop
from scripts.balloon_all_correct_protection import all_correct_loss
from scripts.balloon_positive_protection import retention, feasible
from scripts.train_balloon_reviewed_clutter import CLASSES

PROTOCOL = Path('configs/balloon-real-blue-clutter-20261011.json')


def run_path(mode, seed):
    p = read_json(PROTOCOL)
    if mode not in p['arms'] or seed not in p['seeds']:
        raise ValueError('Undeclared arm or seed')
    return Path(f'runs/balloon-real-blue-clutter-{mode}-seed{seed}-20261011')


def calculate(arrays):
    a = arrays
    for key, shape in (('blue_after', (23, 3)), ('clutter_after', (22, 3)), ('real_after', (39, 3))):
        if a[key].shape != shape or not np.isfinite(a[key]).all():
            raise ValueError('Invalid probe array')
    r = dict(retention=retention(a['before'], a['after'], a['labels'], a['ordinary']),
        reviewed_background_correct=int((a['clutter_after'].argmax(1) == 0).sum()),
        new_blue_accepted=int(((a['blue_after'].argmax(1) == 2) & (softmax(a['blue_after'])[:, 2] >= .8)).sum()),
        real_blue_background_correct=int((a['real_after'].argmax(1) == 0).sum()))
    p = read_json(PROTOCOL)
    r['training_gate_pass'] = (feasible(r, p) and r['new_blue_accepted'] >= p['minimum_new_blue_accepted']
                               and r['real_blue_background_correct'] >= p['minimum_real_blue_background_correct'])
    return r


def inputs(mode):
    rows, blue, clutter, real = data(mode)
    ordinary = np.array([r['preserve'] for r in rows], bool)
    if len(rows) != 1655 or ordinary.sum() != 1533:
        raise ValueError('Matched input counts changed')
    x = np.stack([load_crop(r)[0] for r in rows])
    blue_x = np.stack([load_crop(r)[0] for r in blue])
    clutter_x = np.stack([load_crop(dict(r, view=v))[0] for v in ('original', 'blue_surface') for r in clutter])
    real_x = np.stack([load_crop(r)[0] for r in real])
    return rows, np.array([CLASSES.index(r['label']) for r in rows]), ordinary, (x, blue_x, clutter_x, real_x)


def train(mode, seed):
    p, out = read_json(PROTOCOL), run_path(mode, seed)
    if out.exists():
        raise FileExistsError(out)
    if sha256(p['parent']) != p['parent_sha256']:
        raise ValueError('Parent checkpoint changed')
    rows, labels, ordinary, tensors = inputs(mode)
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    keras.utils.set_random_seed(seed)
    model = keras.models.load_model(p['parent'], compile=False)
    if model.count_params() != 7763 or model.output_shape != (None, 3):
        raise ValueError('Architecture changed')
    before = {k: model(x, training=False).numpy() for k, x in zip(
        ('before', 'blue_before', 'clutter_before', 'real_before'), tensors, strict=True)}
    targets = np.column_stack((np.eye(3)[labels], before['before'], ordinary)).astype(np.float32)
    identity = [{k: r[k] for k in ('source_image', 'sha256', 'label', 'preserve', 'view')} for r in rows]
    source_paths = [str(PROTOCOL), str(REVIEW), str(QUEUE / 'review.json'),
        'scripts/blue_false_alarm_data.py', 'scripts/prepare_blue_false_alarm_review.py',
        'scripts/openimages_blue_expansion.py', 'scripts/grid_clutter_data.py',
        'scripts/balloon_all_correct_protection.py', 'scripts/balloon_positive_protection.py',
        'scripts/train_balloon_reviewed_clutter.py', __file__]
    provenance = dict(mode=mode, seed=seed, settings=p, sample_identity=identity,
        parameters=7763, sources_sha256={path: sha256(path) for path in source_paths},
        tensor_sha256={k: hashlib.sha256(x.tobytes()).hexdigest() for k, x in zip(
            ('train', 'blue', 'clutter', 'real'), tensors, strict=True)},
        optimizer_reset=True, final_epoch_only=True, test_evaluated=False, deployment_approved=False)
    out.mkdir(parents=True)
    write_json(out / 'provenance.json', provenance)
    write_json(out / 'status.json', dict(state='running'))
    try:
        weight = np.where(ordinary, 1., p['hard_weight']).astype(np.float32)
        dataset = tf.data.Dataset.from_tensor_slices((tensors[0], targets, weight)).shuffle(len(rows), seed=seed).batch(p['batch_size'])
        options = tf.data.Options()
        options.threading.private_threadpool_size = 2
        dataset = dataset.with_options(options).prefetch(1)
        model.compile(optimizer=keras.optimizers.Adam(p['learning_rate']),
            loss=all_correct_loss(p['protection_weight'], p['protection_probability_cap'], p['parent_acceptance_threshold']))

        class Progress(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if not logs or not all(np.isfinite(v) for v in logs.values()):
                    raise ValueError('Nonfinite metrics')
                write_json(out / 'status.json', dict(state='running', epoch=epoch + 1, metrics=logs))

        model.fit(dataset, epochs=p['epochs'], verbose=2,
            callbacks=[Progress(), keras.callbacks.CSVLogger(str(out / 'epochs.csv'))])
        arrays = dict(before, labels=labels, ordinary=ordinary)
        arrays.update({k: model(x, training=False).numpy() for k, x in zip(
            ('after', 'blue_after', 'clutter_after', 'real_after'), tensors, strict=True)})
        report = calculate(arrays)
        model.save(out / 'student.keras')
        np.savez_compressed(out / 'predictions.npz', **arrays)
        report.update(model_sha256=sha256(out / 'student.keras'), predictions_sha256=sha256(out / 'predictions.npz'),
            provenance_sha256=sha256(out / 'provenance.json'),
            scope='Paired TRAIN-only feasibility; not scene accuracy or Pi readiness', test_evaluated=False, deployment_approved=False)
        if any(sha256(path) != h for path, h in provenance['sources_sha256'].items()):
            raise ValueError('Source changed while training')
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
    args = parser.parse_args()
    train(args.mode, args.seed)
