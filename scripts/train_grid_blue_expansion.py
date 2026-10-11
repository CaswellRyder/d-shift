"""Paired sparse TRAIN localization adaptation, original grouped holdout unchanged."""
import argparse
import hashlib
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from scripts.grid_blue_expansion_data import expected
from scripts.balloon_dense_iou_loss import giou_loss

PROTOCOL = Path('configs/grid-blue-expansion-20261011.json')


def tensor_hash(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def run_path(mode, seed):
    if mode not in ('control', 'admit') or seed not in (47, 48):
        raise ValueError('Unknown arm or seed')
    return Path(f'runs/grid-blue-expansion-{mode}-seed{seed}-20261011')


def train(mode, seed):
    out = run_path(mode, seed)
    if out.exists():
        raise FileExistsError(out)
    protocol = read_json(PROTOCOL)
    parent = Path(protocol['parent'])
    if sha256(parent) != protocol['parent_sha256']:
        raise ValueError('Warm checkpoint changed')
    data = expected(mode)
    probe = expected('admit') if mode == 'control' else data
    x, y, folds = data['images'], data['targets'], data['folds']
    fit, hold = folds != 0, folds == 0
    if (len(x) != 156 or int(hold.sum()) != 16 or data['stats']['extra_blue_boxes'] != 23
            or not np.array_equal(hold, probe['folds'] == 0)
            or not np.array_equal(x[hold], probe['images'][hold])
            or not np.array_equal(y[hold], probe['targets'][hold])):
        raise ValueError('Paired data/holdout contract changed')
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    keras.utils.set_random_seed(seed)
    model = keras.models.load_model(parent, compile=False)
    if model.count_params() != 2342 or model.output_shape != (None, 15, 20, 6):
        raise ValueError('Architecture changed')
    before = model(probe['images'], training=False).numpy()
    parent_predictions = np.load(parent.parent / 'predictions.npy', allow_pickle=False)
    if not np.allclose(before[:129], parent_predictions, rtol=0, atol=1e-5):
        raise ValueError('Parent predictions do not reproduce')
    sources = dict(data['source_sha256'])
    sources.update({str(p): sha256(p) for p in (PROTOCOL, Path(__file__), Path('scripts/balloon_dense_iou_loss.py'))})
    provenance = dict(mode=mode, seed=seed, parent=str(parent), parent_sha256=sha256(parent),
        sources_sha256=sources, protocol_sha256=sha256(PROTOCOL),
        tensors_sha256={k: tensor_hash(v) for k, v in dict(train_images=x, train_targets=y,
            probe_images=probe['images'], probe_targets=probe['targets']).items()},
        records=data['records'], probe_records=probe['records'], folds=folds.tolist(),
        stats=data['stats'], parameters=2342, optimizer_reset=True, fixed_final_epoch=True,
        test_evaluated=False, deployment_approved=False, pi_tested=False)
    out.mkdir(parents=True)
    write_json(out / 'provenance.json', provenance)
    write_json(out / 'status.json', dict(state='running'))
    try:
        model.compile(optimizer=keras.optimizers.Adam(protocol['learning_rate']), loss=giou_loss)
        dataset = tf.data.Dataset.from_tensor_slices((x[fit], y[fit])).shuffle(int(fit.sum()), seed=seed).batch(protocol['batch_size'])
        options = tf.data.Options()
        options.threading.private_threadpool_size = 2
        dataset = dataset.with_options(options).prefetch(1)

        class Progress(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if not logs or not all(np.isfinite(v) for v in logs.values()):
                    raise ValueError('Nonfinite training loss')
                write_json(out / 'status.json', dict(state='running', epoch=epoch + 1, metrics=logs))
                if (epoch + 1) % 40 == 0:
                    print(dict(mode=mode, seed=seed, epoch=epoch + 1, metrics=logs), flush=True)

        model.fit(dataset, epochs=protocol['epochs'], verbose=0,
                  callbacks=[Progress(), keras.callbacks.CSVLogger(str(out / 'epochs.csv'))])
        after = model(probe['images'], training=False).numpy()
        if after.shape != (156, 15, 20, 6) or not np.isfinite(after).all():
            raise ValueError('Invalid outputs')
        np.savez_compressed(out / 'predictions.npz', before=before, after=after)
        model.save(out / 'proposal.keras')
        if any(sha256(p) != h for p, h in sources.items()):
            raise ValueError('Source changed during training')
        fit_values = model(x[fit], training=False).numpy()
        report = dict(training_loss=float(giou_loss(tf.constant(y[fit]), tf.constant(fit_values))),
            grouped_holdout_loss=float(giou_loss(tf.constant(y[hold]), tf.constant(after[hold]))),
            optimizer_iterations=int(model.optimizer.iterations),
            model_sha256=sha256(out / 'proposal.keras'), predictions_sha256=sha256(out / 'predictions.npz'),
            provenance_sha256=sha256(out / 'provenance.json'), fit_frames=int(fit.sum()), holdout_frames=int(hold.sum()),
            scope='Loss diagnostics only. Proposal coverage gate, export, full-scene and Pi checks still required.',
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
