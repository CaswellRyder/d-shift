"""Matched source-resolution negative supervision; fixed checkpoint only."""
import argparse

import numpy as np

from dtr.data import sha256, write_json
from scripts.source_resolved_negative_data import tensors, matched, run_path
from scripts.clutter_round3_feature_data import targets, loss_function
from scripts.public_scene_localizer_data import tensor_hash


def train(arm, seed):
    out = run_path(arm, seed)
    if out.exists():
        raise FileExistsError(out)
    arms = {a: tensors(a, seed) for a in ('control', 'corrected')}
    matched(arms['control'], arms['corrected'])
    d, p = arms[arm], arms[arm]['settings']
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    keras.utils.set_random_seed(seed)
    model = keras.models.load_model(d['parent']['path'], compile=False)
    if model.count_params() != p['parameters'] or model.output_shape != (None, 3):
        raise ValueError('Architecture changed')
    before = model(d['x'], training=False).numpy()
    target = targets(d['y'], before, d['preserve'], d['scene_mask'])
    out.mkdir(parents=True)
    prov = dict(arm=arm, seed=seed, settings=p, parent=d['parent'], corrections=d['corrections'],
        scene_ids=d['scene_ids'], reviewed_negatives=d['reviewed_negatives'],
        positive_records=d['positive_records'], duplicate_public_inputs=d['duplicate_public_inputs'],
        existing_pixel_matches=d['existing_pixel_matches'], sources_sha256=d['sources_sha256'],
        tensor_sha256={k: tensor_hash(d[k]) for k in ('x', 'y', 'preserve', 'scene_mask', 'weights', 'negative_pixels')},
        target_sha256=tensor_hash(target), fresh_optimizer=True, fixed_final_epoch=True,
        parameters=model.count_params(), development_evaluated=False, test_evaluated=False,
        pi_tested=False, deployment_approved=False, qualification_achieved=False)
    write_json(out/'provenance.json', prov)
    write_json(out/'status.json', dict(state='running'))
    try:
        ds = tf.data.Dataset.from_tensor_slices((d['x'], target, d['weights']))
        options = tf.data.Options()
        options.threading.private_threadpool_size = 2
        ds = ds.shuffle(len(d['x']), seed=seed).batch(p['batch_size']).with_options(options).prefetch(1)
        model.compile(optimizer=keras.optimizers.Adam(keras.optimizers.schedules.CosineDecay(
            p['learning_rate'], p['optimizer_steps'], alpha=p['cosine_alpha'])),
            loss=loss_function(p['scene_kl_weight'], p['scene_centered_logit_weight']))

        class Progress(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if not logs or not all(np.isfinite(v) for v in logs.values()):
                    raise ValueError('Nonfinite training')
                write_json(out/'status.json', dict(state='running', epoch=epoch+1, metrics=logs))
                if (epoch+1) % 20 == 0:
                    print(dict(arm=arm, seed=seed, epoch=epoch+1, loss=logs['loss']), flush=True)

        model.fit(ds, epochs=p['epochs'], verbose=0,
            callbacks=[Progress(), keras.callbacks.CSVLogger(str(out/'epochs.csv'))])
        after = model(d['x'], training=False).numpy()
        if not np.isfinite(after).all() or int(model.optimizer.iterations) != p['optimizer_steps']:
            raise ValueError('Incomplete or nonfinite training')
        if any(sha256(path) != digest for path, digest in d['sources_sha256'].items()):
            raise ValueError('Sources changed during fitting')
        model.save(out/'student.keras')
        np.savez_compressed(out/'predictions.npz', before=before, after=after,
            negatives=model(d['negative_pixels'], training=False).numpy(),
            probes=model(d['probes'], training=False).numpy())
        fit_loss = float(np.mean(loss_function(p['scene_kl_weight'], p['scene_centered_logit_weight'])(
            tf.constant(target), tf.constant(after)).numpy()*d['weights']))
        report = dict(fit_loss=fit_loss, optimizer_steps=int(model.optimizer.iterations),
            artifacts_sha256={name: sha256(out/name) for name in
                ('student.keras', 'predictions.npz', 'provenance.json', 'epochs.csv')},
            scope='TRAIN fitting only; checkpoint and complete scene audit still required.')
        write_json(out/'report.json', report)
        write_json(out/'status.json', dict(state='complete'))
        print(dict(arm=arm, seed=seed, fit_loss=fit_loss), flush=True)
    except BaseException as exc:
        write_json(out/'status.json', dict(state='failed', error=str(exc)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('control', 'corrected'), required=True)
    parser.add_argument('--seed', type=int, choices=(47, 48), required=True)
    args = parser.parse_args()
    train(args.arm, args.seed)
