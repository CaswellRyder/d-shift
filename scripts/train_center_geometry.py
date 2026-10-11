"""Train joint center/geometry residuals with original parent weights frozen."""
import argparse
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from scripts.grid_blue_expansion_data import expected
from scripts.train_grid_blue_expansion import tensor_hash
from scripts.balloon_center_geometry_adapter import network, joint_loss, frozen_weights

PROTOCOL = Path('configs/center-geometry-trial-20261011.json')


def run_path(mode, seed):
    if mode not in ('control', 'admit') or seed not in (47, 48):
        raise ValueError('Unknown paired arm or seed')
    return Path(f'runs/center-geometry-{mode}-seed{seed}-20261011')


def train(mode, seed):
    out = run_path(mode, seed)
    if out.exists():
        raise FileExistsError(out)
    settings = read_json(PROTOCOL)
    parent_path = Path(settings['parent'])
    if sha256(parent_path) != settings['parent_sha256']:
        raise ValueError('Warm parent changed')
    data = expected(mode)
    probe = expected('admit') if mode == 'control' else data
    x, y, folds = data['images'], data['targets'], data['folds']
    fit, hold = folds != 0, folds == 0
    if (len(x) != 156 or int(hold.sum()) != 16 or data['stats']['extra_blue_boxes'] != 23
            or not np.array_equal(hold, probe['folds'] == 0)
            or not np.array_equal(x[hold], probe['images'][hold])
            or not np.array_equal(y[hold], probe['targets'][hold])):
        raise ValueError('Paired data or grouped holdout changed')
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    keras.utils.set_random_seed(seed)
    parent = keras.models.load_model(parent_path, compile=False)
    before = parent(probe['images'], training=False).numpy()
    if not np.allclose(before[:129], np.load(parent_path.parent / 'predictions.npy', allow_pickle=False), rtol=0, atol=1e-5):
        raise ValueError('Warm parent predictions changed')
    model = network(parent)
    initial = model(probe['images'], training=False).numpy()
    initial_delta = float(np.max(np.abs(initial - before)))
    if initial_delta > 1e-5:
        raise ValueError('Zero residual must reproduce full parent output')
    frozen = frozen_weights(model)
    sources = dict(data['source_sha256'])
    sources.update({str(p): sha256(p) for p in (PROTOCOL, Path(__file__),
        Path('scripts/balloon_geometry_adapter.py'), Path('scripts/balloon_center_geometry_adapter.py'), Path('scripts/balloon_dense_iou_loss.py'),
        Path('scripts/train_grid_blue_expansion.py'))})
    provenance = dict(mode=mode, seed=seed, parent=str(parent_path), parent_sha256=sha256(parent_path),
        sources_sha256=sources, protocol_sha256=sha256(PROTOCOL),
        tensors_sha256={k: tensor_hash(v) for k, v in dict(train_images=x, train_targets=y,
            probe_images=probe['images'], probe_targets=probe['targets']).items()},
        records=data['records'], probe_records=probe['records'], folds=folds.tolist(), stats=data['stats'],
        parameters=model.count_params(), trainable_parameters=sum(int(np.prod(w.shape)) for w in model.trainable_weights),
        frozen_weights=frozen, initial_delta=initial_delta, initial_parity_frames=len(initial),
        optimizer_reset=True, fixed_final_epoch=True, test_evaluated=False, pi_tested=False, deployment_approved=False)
    out.mkdir(parents=True)
    write_json(out / 'provenance.json', provenance)
    write_json(out / 'status.json', dict(state='running'))
    try:
        model.compile(optimizer=keras.optimizers.Adam(settings['learning_rate']), loss=joint_loss)
        dataset = tf.data.Dataset.from_tensor_slices((x[fit], y[fit])).shuffle(int(fit.sum()), seed=seed).batch(settings['batch_size'])
        options = tf.data.Options()
        options.threading.private_threadpool_size = 2
        dataset = dataset.with_options(options).prefetch(1)

        class Progress(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if not logs or not all(np.isfinite(v) for v in logs.values()):
                    raise ValueError('Nonfinite geometry loss')
                write_json(out / 'status.json', dict(state='running', epoch=epoch + 1, metrics=logs))
                if (epoch + 1) % 80 == 0:
                    print(dict(mode=mode, seed=seed, epoch=epoch + 1, metrics=logs), flush=True)

        model.fit(dataset, epochs=settings['epochs'], verbose=0,
                  callbacks=[Progress(), keras.callbacks.CSVLogger(str(out / 'epochs.csv'))])
        after = model(probe['images'], training=False).numpy()
        if after.shape != (156, 15, 20, 6) or not np.isfinite(after).all():
            raise ValueError('Invalid predictions')
        heat_delta = float(np.max(np.abs(after[..., :2] - before[..., :2])))
        if frozen_weights(model) != frozen:
            raise ValueError('Frozen parent weights changed')
        np.savez_compressed(out / 'predictions.npz', before=before, after=after)
        model.save(out / 'proposal.keras')
        if any(sha256(p) != h for p, h in sources.items()):
            raise ValueError('Source changed during training')
        fit_values = model(x[fit], training=False).numpy()
        report = dict(training_loss=float(joint_loss(tf.constant(y[fit]), tf.constant(fit_values))),
            grouped_holdout_loss=float(joint_loss(tf.constant(y[hold]), tf.constant(after[hold]))),
            optimizer_iterations=int(model.optimizer.iterations), max_heat_delta=heat_delta,
            frozen_weights_unchanged=True, model_sha256=sha256(out / 'proposal.keras'),
            predictions_sha256=sha256(out / 'predictions.npz'), provenance_sha256=sha256(out / 'provenance.json'),
            fit_frames=int(fit.sum()), holdout_frames=int(hold.sum()),
            scope='Joint loss and frozen parent-weight diagnostics only. No full-scene, export or Pi qualification.',
            test_evaluated=False, deployment_approved=False, pi_tested=False)
        write_json(out / 'report.json', report)
        write_json(out / 'status.json', dict(state='complete'))
        print(dict(mode=mode, seed=seed, report=report), flush=True)
    except BaseException as error:
        write_json(out / 'status.json', dict(state='failed', error=str(error)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('control', 'admit'), required=True)
    parser.add_argument('--seed', type=int, choices=(47, 48), required=True)
    args = parser.parse_args()
    train(args.mode, args.seed)
