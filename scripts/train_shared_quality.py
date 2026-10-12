"""Matched IoU regression and valid-localization classification, head only."""
import argparse
import math

import numpy as np

from dtr.data import sha256, write_json
from scripts.shared_quality_data import build, dependencies, run_path
from scripts.shared_proposal_quality import network
from scripts.public_scene_localizer_data import tensor_hash


def train(arm, seed):
    out = run_path(arm, seed)
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
    if model.count_params() != 281:
        raise ValueError('Architecture changed')
    initial_hashes = [tensor_hash(v.numpy()) for v in model.weights]
    steps = math.ceil(len(data['x'])/p['batch_size'])*p['epochs']
    target = data['y'] if arm == 'iou' else (data['y'] >= .5).astype(np.float32)
    loss = 'mean_squared_error' if arm == 'iou' else 'binary_crossentropy'
    model.compile(optimizer=keras.optimizers.Adam(keras.optimizers.schedules.CosineDecay(
        p['learning_rate'], steps, alpha=p['cosine_alpha'])), loss=loss)
    sources = dependencies()
    out.mkdir(parents=True)
    np.savez(out/'training.npz', **{k: data[k] for k in ('x', 'y', 'normalized', 'mean', 'scale', 'weights')}, target=target)
    write_json(out/'provenance.json', dict(arm=arm, seed=seed, settings=p, sources_sha256=sources,
        parent_sha256=data['parent_sha256'], identities=data['identities'], mser_sources=data['mser_sources'],
        complete_sources=data['complete_sources'], initial_weights_sha256=initial_hashes, optimizer_steps=steps,
        loss=loss, learned_rows=data['learned_rows'], fixed_final_epoch=True, frozen_parent=True,
        original_holdout_evaluated=False, development_evaluated=False, test_evaluated=False,
        deployment_approved=False, qualification_achieved=False))
    write_json(out/'status.json', dict(state='running'))
    try:
        ds = tf.data.Dataset.from_tensor_slices((data['normalized'], target, data['weights']))
        ds = ds.shuffle(len(target), seed=seed).batch(p['batch_size'])
        options = tf.data.Options()
        options.threading.private_threadpool_size = 2
        ds = ds.with_options(options).prefetch(1)

        class Progress(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if not logs or not all(np.isfinite(v) for v in logs.values()):
                    raise ValueError('Nonfinite training')
                write_json(out/'status.json', dict(state='running', epoch=epoch+1, metrics=logs))
                if (epoch+1) % 50 == 0:
                    print(dict(arm=arm, seed=seed, epoch=epoch+1, loss=logs['loss']), flush=True)

        model.fit(ds, epochs=p['epochs'], verbose=0,
            callbacks=[Progress(), keras.callbacks.CSVLogger(str(out/'epochs.csv'))])
        if int(model.optimizer.iterations) != steps or any(sha256(path) != h for path, h in sources.items()):
            raise ValueError('Schedule or dependencies changed')
        pred = model(data['normalized'], training=False).numpy()
        if pred.shape != target.shape or not np.isfinite(pred).all():
            raise ValueError('Invalid final outputs')
        model.save(out/'quality.keras')
        np.save(out/'predictions.npy', pred, allow_pickle=False)
        # model.evaluate uses exactly the compiled per-sample weighted objective.
        evaluated = float(model.evaluate(data['normalized'], target, sample_weight=data['weights'], verbose=0))
        report = dict(fit_rows=len(target), learned_rows=data['learned_rows'], mser_rows=len(target)-data['learned_rows'],
            mser_sources=len(data['mser_sources']), good_boxes=int((data['y'] >= .5).sum()),
            bad_boxes=int((data['y'] < .5).sum()), parameters=281, optimizer_steps=steps, weighted_loss=evaluated,
            artifacts_sha256={name: sha256(out/name) for name in ('quality.keras', 'training.npz', 'predictions.npy', 'provenance.json', 'epochs.csv')},
            scope='TRAIN fitting only; no scene accuracy or deployment qualification.')
        write_json(out/'report.json', report)
        write_json(out/'status.json', dict(state='complete'))
        print(dict(arm=arm, seed=seed, rows=len(target), loss=evaluated), flush=True)
    except BaseException as exc:
        write_json(out/'status.json', dict(state='failed', error=str(exc)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', required=True, choices=('iou', 'valid'))
    parser.add_argument('--seed', required=True, type=int, choices=(47, 48))
    args = parser.parse_args()
    train(args.arm, args.seed)
