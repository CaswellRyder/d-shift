"""Paired fixed-duration whole-frame crop augmentation; no development access."""
import argparse
import math
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from scripts.edge_crop_localizer_data import PROTOCOL, inputs


def run_path(arm, seed):
    if arm not in ('control', 'edge_crop') or seed not in (47, 48):
        raise ValueError('Unknown paired arm or seed')
    return Path(f'runs/edge-crop-localizer-{arm}-seed{seed}-20261011')


def train(arm, seed):
    root = run_path(arm, seed)
    if root.exists():
        raise FileExistsError(root)
    p, data = read_json(PROTOCOL), inputs(arm)
    if sha256(p['parent']) != p['parent_sha256']:
        raise ValueError('Current parent changed')
    parent_root = Path(p['parent']).parent
    prior = read_json(parent_root/'provenance.json')
    if (prior['model_sha256'] != p['parent_sha256']
            or sha256(parent_root/'predictions.npy') != prior['predictions_sha256']
            or any(sha256(path) != h for path, h in prior['sources_sha256'].items())):
        raise ValueError('Parent training provenance changed')
    x, y, folds = data['images'], data['targets'], data['folds']
    fit, hold = folds != 0, data['original_folds'] == 0
    if len(x) != 457 or fit.sum() != 441 or hold.sum() != 16:
        raise ValueError('Paired frame inventory changed')
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    from scripts.balloon_dense_iou_loss import giou_loss
    keras.utils.set_random_seed(seed)
    model = keras.models.load_model(p['parent'], compile=False)
    before = model(data['original_images'], training=False).numpy()
    parent_delta = float(np.max(np.abs(before-np.load(parent_root/'predictions.npy', allow_pickle=False))))
    if parent_delta > 1e-5 or model.count_params() != 2342 or model.output_shape != (None, 15, 20, 6):
        raise ValueError('Parent output or architecture changed')
    steps = math.ceil(int(fit.sum())/p['batch_size'])*p['epochs']
    model.compile(optimizer=keras.optimizers.Adam(keras.optimizers.schedules.CosineDecay(
        p['learning_rate'], steps, alpha=p['cosine_alpha'])), loss=giou_loss)
    sources = {**data['source_sha256'], __file__: sha256(__file__), str(parent_root/'provenance.json'): sha256(parent_root/'provenance.json'),
        'scripts/balloon_dense_iou_loss.py': sha256('scripts/balloon_dense_iou_loss.py')}
    provenance = dict(arm=arm, seed=seed, settings=p, sources_sha256=sources, tensor_sha256=data['hashes'],
        folds=folds.tolist(), records=data['records'], original_groups=data['groups'], extra_frames=data['extra'],
        fit_frames=int(fit.sum()), heldout_frames=int(hold.sum()), parameters=model.count_params(),
        optimizer_steps=steps, fresh_optimizer=True, fixed_final_epoch=True, parent_output_max_delta=parent_delta,
        clipped_visible_augmented_boxes=data['clipped_visible_boxes'],
        test_evaluated=False, development_evaluated=False, deployment_approved=False, pi_tested=False)
    root.mkdir(parents=True)
    write_json(root/'provenance.json', provenance)
    write_json(root/'status.json', dict(state='running'))
    try:
        ds = tf.data.Dataset.from_tensor_slices((x[fit], y[fit])).shuffle(int(fit.sum()), seed=seed).batch(p['batch_size'])
        options = tf.data.Options()
        options.threading.private_threadpool_size = 2
        ds = ds.with_options(options).prefetch(1)

        class Progress(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if not logs or not all(np.isfinite(v) for v in logs.values()):
                    raise ValueError('Nonfinite training')
                write_json(root/'status.json', dict(state='running', epoch=epoch+1, metrics=logs))
                if (epoch+1) % 20 == 0:
                    print(dict(arm=arm, seed=seed, epoch=epoch+1, metrics=logs), flush=True)

        model.fit(ds, epochs=p['epochs'], verbose=0,
            callbacks=[Progress(), keras.callbacks.CSVLogger(str(root/'epochs.csv'))])
        if int(model.optimizer.iterations) != steps:
            raise ValueError('Training step count changed')
        after = model(data['original_images'], training=False).numpy()
        if after.shape != before.shape or not np.isfinite(after).all():
            raise ValueError('Invalid original-view outputs')
        np.savez_compressed(root/'predictions.npz', before=before, after=after)
        model.save(root/'proposal.keras')
        if any(sha256(path) != h for path, h in sources.items()):
            raise ValueError('Sources changed during training')
        original_fit = data['original_folds'] != 0
        report = dict(provenance_sha256=sha256(root/'provenance.json'), model_sha256=sha256(root/'proposal.keras'),
            predictions_sha256=sha256(root/'predictions.npz'), optimizer_steps=int(model.optimizer.iterations),
            original_fit_loss=float(giou_loss(tf.constant(data['original_targets'][original_fit]), tf.constant(after[original_fit]))),
            original_heldout_loss=float(giou_loss(tf.constant(data['original_targets'][hold]), tf.constant(after[hold]))),
            scope='Training diagnostics on original views, not detection accuracy or independent qualification.')
        write_json(root/'report.json', report)
        write_json(root/'status.json', dict(state='complete'))
        print(dict(arm=arm, seed=seed, **report), flush=True)
    except BaseException as exc:
        write_json(root/'status.json', dict(state='failed', error=str(exc)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('control', 'edge_crop'), required=True)
    parser.add_argument('--seed', type=int, choices=(47, 48), required=True)
    args = parser.parse_args()
    train(args.arm, args.seed)
