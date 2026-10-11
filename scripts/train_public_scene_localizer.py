"""Fixed-duration matched retraining of existing2342parameter localization network."""
import argparse
import math

import numpy as np

from dtr.data import sha256, write_json
from scripts.public_scene_localizer_data import tensors, matched, tensor_hash, dependencies, run_path


def train(arm, seed):
    out = run_path(arm, seed)
    if out.exists():
        raise FileExistsError(out)
    arms = {a: tensors(a) for a in ('control', 'scenes')}
    matched(arms['control'], arms['scenes'])
    data = arms[arm]
    x, y, p = data['x'], data['y'], data['settings']
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    from scripts.balloon_dense_iou_loss import giou_loss
    keras.utils.set_random_seed(seed)
    model = keras.models.load_model(p['parent'], compile=False)
    if model.count_params() != p['parameters'] or model.output_shape != (None, 15, 20, 6):
        raise ValueError('Architecture changed')
    if any(isinstance(layer, (keras.layers.BatchNormalization, keras.layers.Dropout)) for layer in model.layers):
        raise ValueError('Control inputs must not influence hidden normalization or dropout state')
    steps = math.ceil(len(x)/p['batch_size'])*p['epochs']
    if steps != p['optimizer_steps']:
        raise ValueError('Optimizer schedule mismatch')
    sources = dependencies()
    model.compile(optimizer=keras.optimizers.Adam(keras.optimizers.schedules.CosineDecay(
        p['learning_rate'], steps, alpha=p['cosine_alpha'])), loss=giou_loss)
    out.mkdir(parents=True)
    provenance = dict(arm=arm, seed=seed, sources_sha256=sources,
        inputs_sha256=tensor_hash(x), targets_sha256=tensor_hash(y),
        **{k: v for k, v in data.items() if k not in ('x', 'y', 'panels')},
        fresh_optimizer=True, fixed_final_checkpoint=True, fit_frames=len(x), excluded_original_holdout_frames=16,
        original_holdout_evaluated=False, development_evaluated=False, test_evaluated=False,
        deployment_approved=False, qualification_achieved=False)
    write_json(out/'provenance.json', provenance)
    write_json(out/'status.json', dict(state='running'))
    try:
        ds = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(len(x), seed=seed).batch(p['batch_size'])
        options = tf.data.Options()
        options.threading.private_threadpool_size = 2
        ds = ds.with_options(options).prefetch(1)

        class Progress(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if not logs or not all(np.isfinite(v) for v in logs.values()):
                    raise ValueError('Nonfinite training')
                write_json(out/'status.json', dict(state='running', epoch=epoch+1, metrics=logs))
                if (epoch+1) % 25 == 0:
                    print(dict(arm=arm, seed=seed, epoch=epoch+1, loss=logs['loss']), flush=True)

        model.fit(ds, epochs=p['epochs'], verbose=0,
            callbacks=[Progress(), keras.callbacks.CSVLogger(str(out/'epochs.csv'))])
        if int(model.optimizer.iterations) != steps or any(sha256(path) != h for path, h in sources.items()):
            raise ValueError('Wrong optimizer steps or sources changed during training')
        model.save(out/'proposal.keras')
        pred = model(x, training=False).numpy()
        if pred.shape != (139, 15, 20, 6) or not np.isfinite(pred).all():
            raise ValueError('Invalid final predictions')
        np.save(out/'predictions.npy', pred, allow_pickle=False)
        report = dict(model_sha256=sha256(out/'proposal.keras'), predictions_sha256=sha256(out/'predictions.npy'),
            provenance_sha256=sha256(out/'provenance.json'), optimizer_steps=int(model.optimizer.iterations),
            fit_loss=float(giou_loss(tf.constant(y), tf.constant(pred))),
            scope='TRAIN fitting loss only; no detection accuracy or qualification claim.')
        write_json(out/'report.json', report)
        write_json(out/'status.json', dict(state='complete'))
        print(dict(arm=arm, seed=seed, **report), flush=True)
    except BaseException as exc:
        write_json(out/'status.json', dict(state='failed', error=str(exc)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('control', 'scenes'), required=True)
    parser.add_argument('--seed', type=int, choices=(47, 48), required=True)
    args = parser.parse_args()
    train(args.arm, args.seed)
