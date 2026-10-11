"""Intentional TRAIN memorization probe, never an export or deployment candidate."""
from pathlib import Path

import numpy as np

from dtr.data import sha256, write_json
from scripts.public_scene_localizer_data import tensors, tensor_hash, dependencies
from scripts.diagnose_public_scene_localizer_cells import inspect

OUTPUT = Path('runs/public-localizer-memorization-20261011')
REPORT = Path('research/balloon-red-blue-20261008/public-localizer-memorization-20261011.json')


def train():
    if OUTPUT.exists() or REPORT.exists():
        raise FileExistsError('Preserve diagnostic fitting run')
    data = tensors('scenes')
    x, y = data['x'][121:], data['y'][121:]
    if x.shape != (18, 120, 160, 3) or y[..., 8].sum() != 40:
        raise ValueError('Require only18reviewed public TRAIN scenes and40box centers')
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    from scripts.balloon_relative_box_loss import relative_loss
    keras.utils.set_random_seed(47)
    model = keras.models.load_model(data['settings']['parent'], compile=False)
    trainable = sum(int(np.prod(v.shape)) for v in model.trainable_variables)
    if trainable != 2342 or model.count_params() != 2342:
        raise ValueError('Unexpected frozen weights or capacity')
    model.compile(optimizer=keras.optimizers.Adam(keras.optimizers.schedules.CosineDecay(
        .001, 7200, alpha=.1)), loss=relative_loss)
    before = model(x, training=False).numpy()
    inputs = {**dependencies(), **{p: sha256(p) for p in (__file__,
        'scripts/balloon_relative_box_loss.py', 'scripts/diagnose_public_scene_localizer_cells.py')}}
    OUTPUT.mkdir(parents=True)
    provenance = dict(parent=data['settings']['parent'], parent_sha256=sha256(data['settings']['parent']),
        seed=47, epochs=400, batch_size=1, learning_rate=.001, cosine_alpha=.1, optimizer_steps=7200,
        input_sha256=tensor_hash(x), target_sha256=tensor_hash(y), sources_sha256=inputs,
        frames=data['additional_frames'], parameters=2342, trainable_parameters=trainable,
        scope='Intentional overfit diagnostic using only18publicTRAINphotos. Ignores old-scene retention by design. Never a deployable candidate or qualification result.',
        development_evaluated=False, test_evaluated=False, pi_tested=False, deployment_approved=False)
    write_json(OUTPUT/'provenance.json', provenance)
    write_json(OUTPUT/'status.json', dict(state='running'))
    try:
        ds = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(18, seed=47).batch(1)
        options = tf.data.Options()
        options.threading.private_threadpool_size = 2
        ds = ds.with_options(options).prefetch(1)

        class Progress(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if not logs or not all(np.isfinite(v) for v in logs.values()):
                    raise ValueError('Nonfinite fitting')
                write_json(OUTPUT/'status.json', dict(state='running', epoch=epoch+1, metrics=logs))
                if (epoch+1) % 100 == 0:
                    print(dict(epoch=epoch+1, loss=logs['loss']), flush=True)

        model.fit(ds, epochs=400, verbose=0,
            callbacks=[Progress(), keras.callbacks.CSVLogger(str(OUTPUT/'epochs.csv'))])
        if int(model.optimizer.iterations) != 7200 or any(sha256(p) != h for p, h in inputs.items()):
            raise ValueError('Schedule or source drift')
        model.save(OUTPUT/'diagnostic.keras')
        restored = keras.models.load_model(OUTPUT/'diagnostic.keras', compile=False)
        after = model(x, training=False).numpy()
        np.testing.assert_allclose(after, restored(x, training=False).numpy(), atol=1e-5, rtol=0)
        np.savez_compressed(OUTPUT/'predictions.npz', before=before, after=after)
        rows = {}
        for name, values in (('before', before), ('after', after)):
            records = [dict(f, targets=inspect(v, f['truth'])) for f, v in zip(data['additional_frames'], values, strict=True)]
            targets = [t for r in records for t in r['targets']]
            rows[name] = dict(frames=records, correct_center_boxes=int(sum(t['center_box_iou'] >= .5 for t in targets)),
                confident_center_cells=int(sum(t['center_probability'] >= .1 for t in targets)),
                localized=int(sum(t['first_loss'] == 'localized' for t in targets)),
                loss=float(relative_loss(tf.constant(y), tf.constant(values))))
        write_json(REPORT, dict(results=rows, provenance_sha256=sha256(OUTPUT/'provenance.json'),
            model_sha256=sha256(OUTPUT/'diagnostic.keras'), predictions_sha256=sha256(OUTPUT/'predictions.npz'),
            script_sha256=sha256(__file__), scope=provenance['scope'],
            development_evaluated=False, test_evaluated=False, pi_tested=False,
            deployment_approved=False, qualification_achieved=False))
        write_json(OUTPUT/'status.json', dict(state='complete'))
        print({name: {k: v for k, v in r.items() if k != 'frames'} for name, r in rows.items()}, flush=True)
    except BaseException as exc:
        write_json(OUTPUT/'status.json', dict(state='failed', error=str(exc)))
        raise


if __name__ == '__main__':
    train()
