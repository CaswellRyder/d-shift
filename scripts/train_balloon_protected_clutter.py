"""Paired final-epoch protection trial; stop at TRAIN feasibility before export."""
import argparse
import hashlib
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from scripts.grid_clutter_data import data, load_crop, QUEUE, REVIEW
from scripts.train_balloon_fragments import MANIFEST
from scripts.train_balloon_reviewed_clutter import CLASSES
from scripts.train_balloon_grid_clutter import checked_protocol as previous_protocol
from scripts.balloon_positive_protection import protected_loss, retention, feasible

PROTOCOL = Path('configs/balloon-protected-clutter-20261011.json')


def train(mode, seed, output):
    old_protocol, protocol = previous_protocol(), read_json(PROTOCOL)
    out, parent_path = Path(output), Path(old_protocol['parent'])
    if out.exists():
        raise FileExistsError(out)
    if mode not in protocol['arms'] or seed not in protocol['seeds']:
        raise ValueError('Undeclared arm or seed')
    parent = read_json(parent_path / 'color.fp32.json')
    if parent['source_student_sha256'] != sha256(parent_path / 'color.keras'):
        raise ValueError('Parent checkpoint changed')
    rows, extra = data(mode)
    ordinary = np.array([r['preserve'] for r in rows], bool)
    if len(rows) != 1531 or int(ordinary.sum()) != 1487:
        raise ValueError('Matched data counts changed')
    identity = [{k: r[k] for k in ('source_image', 'sha256', 'label', 'preserve', 'view')} for r in rows]
    unprotected = Path(f'runs/balloon-grid-clutter-{mode}-seed{seed}-20261011')
    reference = read_json(unprotected / 'provenance.json')
    if (identity != [{k: r[k] for k in identity[0]} for r in reference['sample_identity']]
            or any(reference[k] != protocol[k] for k in ('epochs', 'learning_rate', 'batch_size', 'hard_weight'))
            or reference['parent_checkpoint_sha256'] != sha256(parent_path / 'color.keras')):
        raise ValueError('Matched ablation differs beyond protection loss')
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    keras.utils.set_random_seed(seed)
    model = keras.models.load_model(parent_path / 'color.keras', compile=False)
    if model.count_params() != 7763 or model.output_shape != (None, 3):
        raise ValueError('Architecture changed')
    x = np.stack([load_crop(r)[0] for r in rows])
    labels = np.array([CLASSES.index(r['label']) for r in rows])
    before = model(x, training=False).numpy()
    target = np.column_stack((np.eye(3)[labels], before, ordinary)).astype(np.float32)
    probe_rows = [dict(r, view=view) for view in ('original', 'blue_surface') for r in extra]
    probe = np.stack([load_crop(r)[0] for r in probe_rows])
    probe_before = model(probe, training=False).numpy()
    out.mkdir(parents=True)
    sources = (str(PROTOCOL), 'configs/balloon-grid-clutter-trial-20261011.json',
        'scripts/train_balloon_grid_clutter.py', 'scripts/grid_clutter_data.py',
        'scripts/train_balloon_reviewed_clutter.py', 'scripts/balloon_positive_protection.py',
        str(MANIFEST), str(REVIEW), str(QUEUE / 'review.json'), __file__)
    provenance = dict(mode=mode, seed=seed, sources_sha256={p: sha256(p) for p in sources},
        protocol_sha256=sha256(PROTOCOL), parent_checkpoint_sha256=sha256(parent_path / 'color.keras'),
        parent_model_sha256=parent['sha256'], sample_identity=identity, parameters=model.count_params(),
        unprotected_provenance_sha256=sha256(unprotected / 'provenance.json'),
        unprotected_checkpoint_sha256=sha256(unprotected / 'student.keras'),
        input_tensor_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
        settings=protocol, test_evaluated=False, deployment_approved=False)
    write_json(out / 'provenance.json', provenance)
    write_json(out / 'status.json', dict(state='running'))
    try:
        weight = np.where(ordinary, 1., protocol['hard_weight']).astype(np.float32)
        dataset = tf.data.Dataset.from_tensor_slices((x, target, weight)).shuffle(len(x), seed=seed).batch(protocol['batch_size'])
        options = tf.data.Options()
        options.threading.private_threadpool_size = 2
        dataset = dataset.with_options(options).prefetch(1)
        model.compile(optimizer=keras.optimizers.Adam(protocol['learning_rate']),
            loss=protected_loss(protocol['protection_weight'], protocol['protection_probability_cap'],
                                protocol['parent_acceptance_threshold']))

        class Progress(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if not logs or not all(np.isfinite(v) for v in logs.values()):
                    raise ValueError('Nonfinite metrics')
                write_json(out / 'status.json', dict(state='running', epoch=epoch + 1, metrics=logs))

        model.fit(dataset, epochs=protocol['epochs'], verbose=2,
            callbacks=[Progress(), keras.callbacks.CSVLogger(str(out / 'epochs.csv'))])
        model.save(out / 'student.keras')
        after = model(x, training=False).numpy()
        probe_after = model(probe, training=False).numpy()
        reference_model = keras.models.load_model(unprotected / 'student.keras', compile=False)
        reference_logits = reference_model(x, training=False).numpy()
        np.savez_compressed(out / 'predictions.npz', before=before, after=after, labels=labels,
            ordinary=ordinary, probe_before=probe_before, probe_after=probe_after, unprotected=reference_logits)
        report = dict(retention=retention(before, after, labels, ordinary),
            unprotected_retention=retention(before, reference_logits, labels, ordinary),
            reviewed_background_correct=int((probe_after.argmax(1) == 0).sum()), reviewed_probe_rows=len(probe),
            probe_identity=[dict(id=r['id'], view=r['view'], source_sha256=r['sha256']) for r in probe_rows],
            predictions_sha256=sha256(out / 'predictions.npz'), model_sha256=sha256(out / 'student.keras'),
            provenance_sha256=sha256(out / 'provenance.json'),
            scope='TRAIN feasibility and final-epoch ablation only; no export, development or Pi claim',
            test_evaluated=False, deployment_approved=False)
        report['training_gate_pass'] = feasible(report, protocol)
        if any(sha256(p) != h for p, h in provenance['sources_sha256'].items()):
            raise ValueError('Training source changed during run')
        write_json(out / 'report.json', report)
        write_json(out / 'status.json', dict(state='complete'))
        print(dict(mode=mode, seed=seed, retention=report['retention'],
            probe_correct=report['reviewed_background_correct'], training_gate_pass=report['training_gate_pass']))
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
