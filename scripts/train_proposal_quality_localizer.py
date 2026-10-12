"""Fit a tiny candidate-IoU head while retaining exact frozen localizer outputs."""
import argparse
import math

import numpy as np

from dtr.data import sha256, write_json
from scripts.proposal_quality_data import build, dependencies, run_path
from scripts.proposal_quality_features import network
from scripts.public_scene_localizer_data import tensor_hash


def train(seed):
    out = run_path(seed)
    if out.exists():
        raise FileExistsError(out)
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    data = build(seed)
    p = data['settings']
    keras.utils.set_random_seed(seed)
    model = network()
    if model.count_params() != p['trainable_parameters']:
        raise ValueError('Head architecture changed')
    steps = math.ceil(len(data['x'])/p['batch_size'])*p['epochs']
    model.compile(optimizer=keras.optimizers.Adam(keras.optimizers.schedules.CosineDecay(
        p['learning_rate'], steps, alpha=p['cosine_alpha'])), loss='mean_squared_error')
    sources = dependencies()
    out.mkdir(parents=True)
    np.savez(out/'training.npz', **{k: data[k] for k in ('x', 'y', 'normalized', 'mean', 'scale', 'weights')})
    provenance = dict(seed=seed, sources_sha256=sources, settings=p, parent_sha256=data['parent_sha256'],
        inputs_sha256=tensor_hash(data['x']), targets_sha256=tensor_hash(data['y']),
        approved_complete_sources=data['approved_complete_sources'],
        excluded_incomplete_indices=data['excluded_incomplete_indices'],
        identities=data['identities'], source_records=data['records'], parameters=model.count_params(),
        optimizer_steps=steps, fixed_final_epoch=True, frozen_parent=True,
        original_holdout_evaluated=False, development_evaluated=False, test_evaluated=False,
        deployment_approved=False, qualification_achieved=False)
    write_json(out/'provenance.json', provenance)
    write_json(out/'status.json', dict(state='running'))
    try:
        ds = tf.data.Dataset.from_tensor_slices((data['normalized'], data['y'], data['weights']))
        ds = ds.shuffle(len(data['x']), seed=seed).batch(p['batch_size'])
        options = tf.data.Options()
        options.threading.private_threadpool_size = 2
        ds = ds.with_options(options).prefetch(1)

        class Progress(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if not logs or not all(np.isfinite(v) for v in logs.values()):
                    raise ValueError('Nonfinite training')
                write_json(out/'status.json', dict(state='running', epoch=epoch+1, metrics=logs))
                if (epoch+1) % 50 == 0:
                    print(dict(seed=seed, epoch=epoch+1, loss=logs['loss']), flush=True)

        model.fit(ds, epochs=p['epochs'], verbose=0,
            callbacks=[Progress(), keras.callbacks.CSVLogger(str(out/'epochs.csv'))])
        if int(model.optimizer.iterations) != steps or any(sha256(path) != h for path, h in sources.items()):
            raise ValueError('Schedule or sources changed during training')
        pred = model(data['normalized'], training=False).numpy()
        if pred.shape != data['y'].shape or not np.isfinite(pred).all():
            raise ValueError('Invalid final quality outputs')
        model.save(out/'quality.keras')
        np.save(out/'predictions.npy', pred, allow_pickle=False)
        report = dict(fit_proposals=len(pred), complete_sources=len(data['approved_complete_sources']),
            good_boxes=int((data['y'] >= .5).sum()), bad_boxes=int((data['y'] < .5).sum()),
            weighted_mse=float(np.mean((pred[:, 0]-data['y'][:, 0])**2*data['weights'])),
            optimizer_steps=steps, parameters=model.count_params(),
            artifacts_sha256={name: sha256(out/name) for name in ('quality.keras', 'training.npz', 'predictions.npy', 'provenance.json', 'epochs.csv')},
            scope='TRAIN fit only. No detection or hardware qualification.')
        write_json(out/'report.json', report)
        write_json(out/'status.json', dict(state='complete'))
        print(dict(seed=seed, proposals=len(pred), mse=report['weighted_mse']), flush=True)
    except BaseException as exc:
        write_json(out/'status.json', dict(state='failed', error=str(exc)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=(47, 48), required=True)
    train(parser.parse_args().seed)
