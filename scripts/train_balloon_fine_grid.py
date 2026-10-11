"""Paired fine-grid training with unchanged source splits and fixed final epoch."""
import argparse
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from scripts.balloon_fine_grid import PROTOCOL, checked_grid_data, network, loss


def train(seed, output):
    out = Path(output)
    if out.exists():
        raise FileExistsError(out)
    protocol = read_json(PROTOCOL)
    if seed not in protocol['seeds'] or protocol['output_grid'] != [30, 40]:
        raise ValueError('Require fixed paired protocol')
    doc, x, y, folds, groups = checked_grid_data()
    fit, hold = folds != 0, folds == 0
    control = Path(protocol['controls'][str(seed)])
    cp, cr = read_json(control / 'provenance.json'), read_json(control / 'report.json')
    if (read_json(control / 'status.json')['state'] != 'complete'
            or cp['manifest_sha256'] != protocol['manifest_sha256']
            or cp['seed'] != seed or cp['folds'] != folds.tolist()
            or any(cp[k] != protocol[k] for k in ('epochs', 'batch_size', 'learning_rate', 'box_objective'))
            or cp['script_sha256'] != sha256('scripts/train_balloon_dense_proposals.py')
            or cr['provenance_sha256'] != sha256(control / 'provenance.json')
            or cr['model_sha256'] != sha256(control / 'proposal.keras')):
        raise ValueError('Paired control changed')
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    keras.utils.set_random_seed(seed)
    model = network()
    if model.count_params() != 2342 or model.output_shape != (None, 30, 40, 6):
        raise ValueError('Architecture changed')
    out.mkdir(parents=True)
    np.save(out / 'targets.npy', y, allow_pickle=False)
    provenance = dict(manifest_sha256=protocol['manifest_sha256'], seed=seed,
        epochs=protocol['epochs'], batch_size=protocol['batch_size'], learning_rate=protocol['learning_rate'],
        parameters=model.count_params(), folds=folds.tolist(), groups=groups, fixed_final_epoch=True,
        script_sha256=sha256(__file__), protocol_sha256=sha256(PROTOCOL),
        grid_helper_sha256=sha256('scripts/balloon_fine_grid.py'),
        targets_sha256=sha256(out / 'targets.npy'),
        helpers_sha256={p: sha256(p) for p in ('scripts/train_balloon_dense_proposals.py',
            'scripts/balloon_dense_iou_loss.py', 'scripts/balloon_dense_complete.py')},
        control_provenance_sha256=sha256(control / 'provenance.json'),
        input_hw=[120, 160], grid=[30, 40], box_objective='l1_giou',
        omitted_negative_regions=doc['fine_grid_omitted_negatives'],
        preparation_script_sha256=doc['script_sha256'],
        scope='Fine-grid pipeline trial; same reviewed source geometry, input pixels, initial weights and optimizer schedule',
        test_evaluated=False, deployment_approved=False, pi_tested=False)
    write_json(out / 'provenance.json', provenance)
    write_json(out / 'status.json', dict(state='running'))
    try:
        ds = tf.data.Dataset.from_tensor_slices((x[fit], y[fit])).shuffle(int(fit.sum()), seed=seed).batch(protocol['batch_size'])
        options = tf.data.Options()
        options.threading.private_threadpool_size = 2
        ds = ds.with_options(options).prefetch(1)
        model.compile(optimizer=keras.optimizers.Adam(protocol['learning_rate']), loss=loss)

        class Progress(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if not logs or not all(np.isfinite(v) for v in logs.values()):
                    raise ValueError('Nonfinite training')
                write_json(out / 'status.json', dict(state='running', epoch=epoch + 1, metrics=logs))

        model.fit(ds, epochs=protocol['epochs'], verbose=2,
                  callbacks=[Progress(), keras.callbacks.CSVLogger(str(out / 'epochs.csv'))])
        model.save(out / 'proposal.keras')
        values = model(x, training=False).numpy()
        if values.shape != (len(x), 30, 40, 6) or not np.isfinite(values).all():
            raise ValueError('Invalid proposal outputs')
        np.save(out / 'predictions.npy', values, allow_pickle=False)
        report = dict(training_loss=float(loss(tf.constant(y[fit]), tf.constant(values[fit]))),
            grouped_holdout_loss=float(loss(tf.constant(y[hold]), tf.constant(values[hold]))),
            parameters=model.count_params(), fit_frames=int(fit.sum()), holdout_frames=int(hold.sum()),
            model_sha256=sha256(out / 'proposal.keras'), predictions_sha256=sha256(out / 'predictions.npy'),
            provenance_sha256=sha256(out / 'provenance.json'),
            scope='Loss diagnostics, not detection accuracy', test_evaluated=False, deployment_approved=False, pi_tested=False)
        write_json(out / 'report.json', report)
        write_json(out / 'status.json', dict(state='complete'))
        print(report)
    except BaseException as exc:
        write_json(out / 'status.json', dict(state='failed', error=str(exc)))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed', type=int, choices=(47, 48), required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    train(a.seed, a.output)
