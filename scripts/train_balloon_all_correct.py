"""Matched protection-mask ablation from the same admitted80 checkpoint per seed."""
import argparse
import hashlib
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from scripts.audit_balloon_protected_convergence import checked_run
from scripts.balloon_all_correct_protection import all_correct_loss
from scripts.balloon_positive_protection import protected_loss, retention, feasible
from scripts.continue_balloon_protected_clutter import optimizer_evidence, LOSS_PROTOCOL
from scripts.grid_clutter_data import data, load_crop

PROTOCOL = Path('configs/balloon-all-correct-protection-20261011.json')


def train(arm, seed, output):
    out, protocol, loss_config = Path(output), read_json(PROTOCOL), read_json(LOSS_PROTOCOL)
    if out.exists():
        raise FileExistsError(out)
    if arm not in protocol['arms'] or seed not in protocol['seeds']:
        raise ValueError('Unknown loss arm or seed')
    parent, previous, report, arrays = checked_run('admit', seed)
    rows, extra = data('admit')
    x = np.stack([load_crop(r)[0] for r in rows])
    if hashlib.sha256(x.tobytes()).hexdigest() != previous['input_tensor_sha256']:
        raise ValueError('Ablation pixels differ from continuation parent')
    labels, ordinary = arrays['labels'], arrays['ordinary']
    target = np.column_stack((np.eye(3)[labels], arrays['before'], ordinary)).astype(np.float32)
    probe = np.stack([load_crop(dict(r, view=v))[0] for v in ('original', 'blue_surface') for r in extra])
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    keras.utils.set_random_seed(seed)
    options = (loss_config['protection_weight'], loss_config['protection_probability_cap'], loss_config['parent_acceptance_threshold'])
    model = keras.models.load_model(parent / 'student.keras', custom_objects={'loss': protected_loss(*options)})
    if model.count_params() != 7763 or model.output_shape != (None, 3):
        raise ValueError('Architecture changed')
    restored = optimizer_evidence(model, 7680)
    state = [v.numpy().copy() for v in model.optimizer.variables]
    model.compile(optimizer=model.optimizer, loss=all_correct_loss(*options) if arm == 'all_correct' else protected_loss(*options))
    if any(not np.array_equal(a, b.numpy()) for a, b in zip(state, model.optimizer.variables, strict=True)):
        raise ValueError('Recompiling loss reset optimizer state')
    if not np.allclose(model(x, training=False).numpy(), arrays['after'], rtol=0, atol=1e-5):
        raise ValueError('Parent predictions do not reproduce')
    out.mkdir(parents=True)
    sources = (str(PROTOCOL), str(LOSS_PROTOCOL), 'scripts/audit_balloon_protected_convergence.py',
        'scripts/balloon_all_correct_protection.py', 'scripts/balloon_positive_protection.py',
        'scripts/continue_balloon_protected_clutter.py', 'scripts/grid_clutter_data.py', __file__)
    provenance = dict(arm=arm, seed=seed, parent=str(parent), parent_model_sha256=report['model_sha256'],
        parent_provenance_sha256=report['provenance_sha256'], parent_predictions_sha256=report['predictions_sha256'],
        input_tensor_sha256=previous['input_tensor_sha256'], protocol_sha256=sha256(PROTOCOL),
        loss_protocol_sha256=sha256(LOSS_PROTOCOL), sources_sha256={p: sha256(p) for p in sources},
        optimizer_restored=restored, optimizer_slots_preserved_on_recompile=True,
        parameters=model.count_params(), initial_epoch=80, final_epoch=104,
        original_parent_targets_preserved=True, dataset_shuffle_restarted=True, fixed_final_epoch=True,
        test_evaluated=False, deployment_approved=False)
    write_json(out / 'provenance.json', provenance)
    write_json(out / 'status.json', dict(state='running'))
    try:
        weights = np.where(ordinary, 1., loss_config['hard_weight']).astype(np.float32)
        dataset = tf.data.Dataset.from_tensor_slices((x, target, weights)).shuffle(len(x), seed=seed).batch(16)
        dataset_options = tf.data.Options()
        dataset_options.threading.private_threadpool_size = 2
        dataset = dataset.with_options(dataset_options).prefetch(1)

        class Progress(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if not logs or not all(np.isfinite(v) for v in logs.values()):
                    raise ValueError('Nonfinite metrics')
                write_json(out / 'status.json', dict(state='running', epoch=epoch + 1, metrics=logs))

        model.fit(dataset, initial_epoch=80, epochs=104, verbose=2,
            callbacks=[Progress(), keras.callbacks.CSVLogger(str(out / 'epochs.csv'))])
        final_optimizer = optimizer_evidence(model, 9984)
        model.save(out / 'student.keras')
        after, probe_after = model(x, training=False).numpy(), model(probe, training=False).numpy()
        if not np.isfinite(after).all() or not np.isfinite(probe_after).all():
            raise ValueError('Nonfinite outputs')
        np.savez_compressed(out / 'predictions.npz', **dict(arrays, after=after, probe_after=probe_after))
        final = dict(retention=retention(arrays['before'], after, labels, ordinary),
            reviewed_background_correct=int((probe_after.argmax(1) == 0).sum()), reviewed_probe_rows=22,
            probe_identity=report['probe_identity'], optimizer_final=final_optimizer,
            model_sha256=sha256(out / 'student.keras'), predictions_sha256=sha256(out / 'predictions.npz'),
            provenance_sha256=sha256(out / 'provenance.json'),
            scope='Matched TRAIN mask ablation, fixed final104 only; not held-out accuracy',
            test_evaluated=False, deployment_approved=False)
        final['training_gate_pass'] = feasible(final, loss_config)
        if any(sha256(p) != h for p, h in provenance['sources_sha256'].items()):
            raise ValueError('Sources changed while training')
        write_json(out / 'report.json', final)
        write_json(out / 'status.json', dict(state='complete'))
        print(dict(arm=arm, seed=seed, retention=final['retention'],
            background_correct=final['reviewed_background_correct'], training_gate_pass=final['training_gate_pass']))
    except BaseException as exc:
        write_json(out / 'status.json', dict(state='failed', error=str(exc)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('accepted', 'all_correct'), required=True)
    parser.add_argument('--seed', type=int, choices=(47, 48), required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    train(args.arm, args.seed, args.output)
