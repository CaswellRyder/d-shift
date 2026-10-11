"""Matched complete-negative supervision, fixed classifier and localizer architecture."""
import argparse
import hashlib
import math

import numpy as np

from dtr.data import read_json, sha256, write_json
from scripts.blue_scene_localizer_data import PROTOCOL, data, dependencies, run_path


def train(arm, seed):
    out = run_path(arm, seed)
    if out.exists():
        raise FileExistsError(out)
    p = read_json(PROTOCOL)
    x, y, folds, identities, groups = data(arm)
    fit, hold = folds != 0, folds == 0
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    from scripts.balloon_dense_iou_loss import giou_loss
    keras.utils.set_random_seed(seed)
    model = keras.models.load_model(p['parent'], compile=False)
    if (model.count_params() != 2342 or model.output_shape != (None, 15, 20, 6)
            or any(isinstance(layer, (keras.layers.BatchNormalization, keras.layers.Dropout)) for layer in model.layers)):
        raise ValueError('Architecture or batch coupling changed')
    steps = math.ceil(int(fit.sum()) / p['batch_size']) * p['epochs']
    model.compile(optimizer=keras.optimizers.Adam(keras.optimizers.schedules.CosineDecay(
        p['learning_rate'], steps, alpha=p['cosine_alpha'])), loss=giou_loss)
    out.mkdir(parents=True)
    np.save(out/'targets.npy', y, allow_pickle=False)
    provenance = dict(arm=arm, seed=seed, protocol_sha256=sha256(PROTOCOL),
        script_sha256=sha256(__file__), helpers_sha256=dependencies(),
        parent_sha256=sha256(p['parent']), targets_sha256=sha256(out/'targets.npy'),
        inputs_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
        folds=folds.tolist(), original_groups=groups, additional_frames=identities,
        fit_frames=int(fit.sum()), holdout_frames=int(hold.sum()), optimizer_steps=steps,
        epochs=p['epochs'], fresh_optimizer=True, fixed_final_checkpoint=True,
        parameters=model.count_params(), test_evaluated=False, deployment_approved=False, pi_tested=False)
    write_json(out/'provenance.json', provenance)
    write_json(out/'status.json', dict(state='running'))
    try:
        ds = tf.data.Dataset.from_tensor_slices((x[fit], y[fit])).shuffle(int(fit.sum()), seed=seed).batch(p['batch_size'])
        options = tf.data.Options()
        options.threading.private_threadpool_size = 2
        ds = ds.with_options(options).prefetch(1)

        class Progress(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if not logs or not all(np.isfinite(v) for v in logs.values()):
                    raise ValueError('Nonfinite training')
                write_json(out/'status.json', dict(state='running', epoch=epoch+1, metrics=logs))

        model.fit(ds, epochs=p['epochs'], verbose=2,
            callbacks=[Progress(), keras.callbacks.CSVLogger(str(out/'epochs.csv'))])
        if int(model.optimizer.iterations) != steps:
            raise ValueError('Wrong optimizer step count')
        model.save(out/'proposal.keras')
        pred = model(x, training=False).numpy()
        if pred.shape != (len(x), 15, 20, 6) or not np.isfinite(pred).all():
            raise ValueError('Invalid final predictions')
        np.save(out/'predictions.npy', pred, allow_pickle=False)
        report = dict(model_sha256=sha256(out/'proposal.keras'),
            predictions_sha256=sha256(out/'predictions.npy'), provenance_sha256=sha256(out/'provenance.json'),
            optimizer_steps=int(model.optimizer.iterations),
            fit_loss=float(giou_loss(tf.constant(y[fit]), tf.constant(pred[fit]))),
            holdout_loss=float(giou_loss(tf.constant(y[hold]), tf.constant(pred[hold]))),
            scope='Loss diagnostics, not object detection accuracy or qualification')
        write_json(out/'report.json', report)
        write_json(out/'status.json', dict(state='complete'))
        print(report, flush=True)
    except BaseException as exc:
        write_json(out/'status.json', dict(state='failed', error=str(exc)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('control', 'background'), required=True)
    parser.add_argument('--seed', type=int, choices=(47, 48), required=True)
    args = parser.parse_args()
    train(args.arm, args.seed)
