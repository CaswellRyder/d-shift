"""Bounded TRAIN-only relative-size loss trial; no inference graph changes."""
import argparse
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from scripts.balloon_dense_iou_loss import giou_loss
from scripts.train_balloon_dense_proposals import checked_data, network

PROTOCOL = Path('configs/balloon-relative-size-trial-20261011.json')


def relative_size_penalty(target, prediction):
    """Equal multiplicative width/height errors receive equal penalties.

    Select valid cells first so unlabelled geometry has no gradient. The fixed
    epsilon only protects log(0); it is not a postprocessing size constraint.
    """
    import tensorflow as tf
    valid = target[..., 8] > 0
    truth = tf.boolean_mask(target[..., 6:8], valid)
    size = tf.boolean_mask(prediction[..., 4:6], valid)
    delta = tf.math.log(tf.maximum(size, 1e-4)) - tf.math.log(tf.maximum(truth, 1e-4))
    error = tf.abs(delta)
    smooth = tf.where(error < 1., .5 * tf.square(error), error - .5)
    return tf.reduce_sum(smooth) / tf.maximum(tf.cast(tf.size(smooth), tf.float32), 1.)


def relative_size_loss(target, prediction):
    return giou_loss(target, prediction) + relative_size_penalty(target, prediction)


def train(seed, output):
    out = Path(output)
    if out.exists():
        raise FileExistsError(out)
    protocol = read_json(PROTOCOL)
    if seed not in protocol['seeds']:
        raise ValueError('Require fixed paired seed')
    root = Path(protocol['data'])
    if sha256(root / 'manifest.json') != protocol['manifest_sha256']:
        raise ValueError('Training data changed')
    if (protocol['size_weight'] != 1. or protocol['size_epsilon'] != 1e-4
            or protocol['epochs'] != 80 or protocol['batch_size'] != 8
            or protocol['learning_rate'] != .001):
        raise ValueError('Fixed loss protocol changed')
    doc, x, y, folds, groups = checked_data(root)
    fit, hold = folds != 0, folds == 0
    control = Path(protocol['controls'][str(seed)])
    cp, cr = read_json(control / 'provenance.json'), read_json(control / 'report.json')
    if (read_json(control / 'status.json')['state'] != 'complete'
            or cp['manifest_sha256'] != protocol['manifest_sha256']
            or cp['seed'] != seed or cp['folds'] != folds.tolist()
            or any(cp[k] != protocol[k] for k in ('epochs', 'batch_size', 'learning_rate'))
            or cp['box_objective'] != 'l1_giou'
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
    out.mkdir(parents=True)
    provenance = dict(manifest_sha256=protocol['manifest_sha256'], seed=seed,
        epochs=80, batch_size=8, learning_rate=.001, parameters=model.count_params(),
        folds=folds.tolist(), groups=groups, fixed_final_epoch=True,
        script_sha256=sha256(__file__), protocol_sha256=sha256(PROTOCOL),
        architecture_helper_sha256=sha256('scripts/train_balloon_dense_proposals.py'),
        loss_helper_sha256=sha256('scripts/balloon_dense_iou_loss.py'),
        control_provenance_sha256=sha256(control / 'provenance.json'),
        box_objective='l1_giou_relative_size', size_weight=1., size_epsilon=1e-4,
        preparation_script_sha256=doc['script_sha256'],
        scope='Only objective changes; same data, architecture, seed and optimizer schedule',
        test_evaluated=False, deployment_approved=False, pi_tested=False)
    write_json(out / 'provenance.json', provenance)
    write_json(out / 'status.json', dict(state='running'))
    try:
        ds = tf.data.Dataset.from_tensor_slices((x[fit], y[fit])).shuffle(int(fit.sum()), seed=seed).batch(8)
        options = tf.data.Options()
        options.threading.private_threadpool_size = 2
        ds = ds.with_options(options).prefetch(1)
        model.compile(optimizer=keras.optimizers.Adam(.001), loss=relative_size_loss)

        class Progress(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if not logs or not all(np.isfinite(v) for v in logs.values()):
                    raise ValueError('Nonfinite training')
                write_json(out / 'status.json', dict(state='running', epoch=epoch + 1, metrics=logs))

        model.fit(ds, epochs=80, verbose=2, callbacks=[Progress(), keras.callbacks.CSVLogger(str(out / 'epochs.csv'))])
        model.save(out / 'proposal.keras')
        values = model(x, training=False).numpy()
        if values.shape != (len(x), 15, 20, 6) or not np.isfinite(values).all():
            raise ValueError('Invalid proposal outputs')
        np.save(out / 'predictions.npy', values, allow_pickle=False)
        metrics = {name: dict(combined=float(relative_size_loss(tf.constant(y[mask]), tf.constant(values[mask]))),
                    relative_size=float(relative_size_penalty(tf.constant(y[mask]), tf.constant(values[mask]))),
                    original_objective=float(giou_loss(tf.constant(y[mask]), tf.constant(values[mask]))))
                   for name, mask in (('fit', fit), ('grouped_holdout', hold))}
        report = dict(metrics=metrics, parameters=model.count_params(), fit_frames=int(fit.sum()),
            holdout_frames=int(hold.sum()), model_sha256=sha256(out / 'proposal.keras'),
            predictions_sha256=sha256(out / 'predictions.npy'),
            provenance_sha256=sha256(out / 'provenance.json'),
            scope='Loss diagnostics, not detection accuracy',
            test_evaluated=False, deployment_approved=False, pi_tested=False)
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
