"""Fixed-duration optimizer-state continuation; never fit reserved development frames."""
import argparse
import math
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json

PROTOCOL = Path('configs/balloon-grid-convergence-20261011.json')


def optimizer_evidence(model, expected_iterations):
    variables = model.optimizer.variables
    if (int(model.optimizer.iterations) != expected_iterations
            or not np.isclose(float(model.optimizer.learning_rate), .001, rtol=0, atol=1e-9)
            or len(variables) != 2 + 2 * len(model.trainable_variables)
            or not all(np.isfinite(v.numpy()).all() for v in variables)
            or not any(np.any(v.numpy()) for v in variables[2:])):
        raise ValueError('Adam state was not restored correctly')
    return dict(iterations=int(model.optimizer.iterations), learning_rate=float(model.optimizer.learning_rate),
                variables=len(variables), trainable_tensors=len(model.trainable_variables), finite_slots=True,
                nonzero_slots=any(np.any(v.numpy()) for v in variables[2:]))


def source_data(arm):
    protocol = read_json(PROTOCOL)
    from scripts.train_balloon_dense_proposals import checked_data
    from scripts.balloon_fine_grid import checked_grid_data
    root = Path(protocol['data'])
    if sha256(root / 'manifest.json') != protocol['manifest_sha256']:
        raise ValueError('Source dataset changed')
    if arm == 'coarse':
        return checked_data(root)
    if arm == 'context':
        return checked_grid_data()
    raise ValueError('Unknown grid arm')


def train(arm, seed, output):
    out = Path(output)
    if out.exists():
        raise FileExistsError(out)
    protocol = read_json(PROTOCOL)
    if arm not in protocol['arms'] or seed not in protocol['seeds']:
        raise ValueError('Require paired arm and seed')
    doc, x, targets, folds, groups = source_data(arm)
    fit, hold = folds != 0, folds == 0
    parent = Path(f"runs/{protocol['parent_prefix'][arm]}-seed{seed}-20261011")
    pp, pr = read_json(parent / 'provenance.json'), read_json(parent / 'report.json')
    if (read_json(parent / 'status.json')['state'] != 'complete'
            or pp['seed'] != seed or pp['epochs'] != protocol['parent_epochs']
            or pp['batch_size'] != protocol['batch_size'] or pp['learning_rate'] != protocol['learning_rate']
            or pp['folds'] != folds.tolist() or pp['manifest_sha256'] != protocol['manifest_sha256']
            or pr['provenance_sha256'] != sha256(parent / 'provenance.json')
            or pr['model_sha256'] != sha256(parent / 'proposal.keras')
            or pr['predictions_sha256'] != sha256(parent / 'predictions.npy')):
        raise ValueError('Parent identity or schedule changed')
    if arm == 'context':
        from scripts.audit_balloon_context_grid import checked_predictions
        checked_predictions(parent, targets, folds)
    elif pp['script_sha256'] != sha256('scripts/train_balloon_dense_proposals.py'):
        raise ValueError('Coarse parent trainer changed')
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    from scripts.balloon_dense_iou_loss import giou_loss
    from scripts.balloon_fine_grid import loss
    keras.utils.set_random_seed(seed)
    model = keras.models.load_model(parent / 'proposal.keras', custom_objects={'giou_loss': giou_loss, 'loss': loss})
    steps = math.ceil(int(fit.sum()) / protocol['batch_size'])
    restored = optimizer_evidence(model, protocol['parent_epochs'] * steps)
    reproduced = model(x, training=False).numpy()
    if not np.allclose(reproduced, np.load(parent / 'predictions.npy', allow_pickle=False), atol=1e-5, rtol=0):
        raise ValueError('Parent checkpoint does not reproduce outputs')
    out.mkdir(parents=True)
    np.save(out / 'targets.npy', targets, allow_pickle=False)
    provenance = dict(arm=arm, seed=seed, parent=str(parent), parent_model_sha256=pr['model_sha256'],
        parent_provenance_sha256=pr['provenance_sha256'], parent_report_sha256=sha256(parent / 'report.json'),
        manifest_sha256=protocol['manifest_sha256'], epochs=protocol['final_epochs'],
        initial_epoch=protocol['parent_epochs'], batch_size=protocol['batch_size'], learning_rate=protocol['learning_rate'],
        folds=folds.tolist(), groups=groups, targets_sha256=sha256(out / 'targets.npy'),
        script_sha256=sha256(__file__), protocol_sha256=sha256(PROTOCOL),
        helpers_sha256={p: sha256(p) for p in ('scripts/balloon_fine_grid.py', 'scripts/balloon_dense_iou_loss.py',
            'scripts/train_balloon_dense_proposals.py', 'scripts/balloon_context_grid.py')},
        optimizer_restored=restored, dataset_shuffle_restarted=True, fixed_final_epoch=True,
        parameters=model.count_params(), input_shape=list(model.input_shape), output_shape=list(model.output_shape),
        test_evaluated=False, deployment_approved=False, pi_tested=False)
    write_json(out / 'provenance.json', provenance)
    write_json(out / 'status.json', dict(state='running'))
    try:
        ds = tf.data.Dataset.from_tensor_slices((x[fit], targets[fit])).shuffle(int(fit.sum()), seed=seed).batch(protocol['batch_size'])
        options = tf.data.Options()
        options.threading.private_threadpool_size = 2
        ds = ds.with_options(options).prefetch(1)

        class Progress(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if not logs or not all(np.isfinite(v) for v in logs.values()):
                    raise ValueError('Nonfinite training')
                write_json(out / 'status.json', dict(state='running', epoch=epoch + 1, metrics=logs))

        model.fit(ds, initial_epoch=protocol['parent_epochs'], epochs=protocol['final_epochs'], verbose=2,
                  callbacks=[Progress(), keras.callbacks.CSVLogger(str(out / 'epochs.csv'))])
        final_optimizer = optimizer_evidence(model, protocol['final_epochs'] * steps)
        model.save(out / 'proposal.keras')
        values = model(x, training=False).numpy()
        if values.shape != reproduced.shape or not np.isfinite(values).all():
            raise ValueError('Invalid outputs')
        np.save(out / 'predictions.npy', values, allow_pickle=False)
        loss_fn = giou_loss if arm == 'coarse' else loss
        report = dict(training_loss=float(loss_fn(tf.constant(targets[fit]), tf.constant(values[fit]))),
            grouped_holdout_loss=float(loss_fn(tf.constant(targets[hold]), tf.constant(values[hold]))),
            model_sha256=sha256(out / 'proposal.keras'), predictions_sha256=sha256(out / 'predictions.npy'),
            provenance_sha256=sha256(out / 'provenance.json'), optimizer_final=final_optimizer,
            fit_frames=int(fit.sum()), holdout_frames=int(hold.sum()),
            scope='Loss diagnostics only, not full-scene accuracy or Pi readiness',
            test_evaluated=False, deployment_approved=False, pi_tested=False)
        write_json(out / 'report.json', report)
        write_json(out / 'status.json', dict(state='complete'))
        print(report)
    except BaseException as exc:
        write_json(out / 'status.json', dict(state='failed', error=str(exc)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('coarse', 'context'), required=True)
    parser.add_argument('--seed', type=int, choices=(47, 48), required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    train(args.arm, args.seed, args.output)
