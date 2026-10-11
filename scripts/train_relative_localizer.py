"""Matched objective-only retraining; reuse verified existing exact-data controls."""
import argparse
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from scripts.public_scene_localizer_data import tensors, tensor_hash, dependencies

PROTOCOL = Path('configs/relative-localizer-20261011.json')


def settings():
    p = read_json(PROTOCOL)
    if (p['training_approved'] is not True or sha256(p['source_protocol']) != p['source_protocol_sha256']
            or sha256(p['source_audit']) != p['source_audit_sha256']
            or [p[k] for k in ('relative_weight', 'minimum_center_denominator_pixels',
                'minimum_size_log_pixels', 'smooth_l1_transition')] != [2., 4., 1., 1.]
            or any(p[k] is not False for k in ('development_evaluated', 'test_evaluated', 'deployment_approved'))):
        raise ValueError('Objective, sources or scope changed')
    return p


def sources():
    return {**dependencies(), **{str(p): sha256(p) for p in
        (PROTOCOL, __file__, 'scripts/balloon_relative_box_loss.py', 'scripts/audit_public_scene_localizer.py')}}


def run_path(seed):
    if seed not in (47, 48):
        raise ValueError('Undeclared seed')
    return Path(f'runs/relative-localizer-seed{seed}-20261011')


def train(seed):
    out = run_path(seed)
    if out.exists():
        raise FileExistsError(out)
    trial, data = settings(), tensors('scenes')
    x, y, p = data['x'], data['y'], data['settings']
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    from scripts.balloon_relative_box_loss import relative_loss, relative_penalty
    from scripts.balloon_dense_iou_loss import giou_loss
    from scripts.audit_public_scene_localizer import verify
    parent = keras.models.load_model(p['parent'], compile=False)
    _, control = verify('scenes', seed, data, parent)
    if control['run'] != trial['matched_controls'][str(seed)]:
        raise ValueError('Mismatched control')
    keras.utils.set_random_seed(seed)
    model = keras.models.load_model(p['parent'], compile=False)
    model.compile(optimizer=keras.optimizers.Adam(keras.optimizers.schedules.CosineDecay(
        p['learning_rate'], p['optimizer_steps'], alpha=p['cosine_alpha'])), loss=relative_loss)
    out.mkdir(parents=True)
    input_sources = sources()
    provenance = dict(seed=seed, settings=trial, base_settings=p, sources_sha256=input_sources,
        inputs_sha256=tensor_hash(x), targets_sha256=tensor_hash(y), verified_control=control,
        fresh_optimizer=True, fixed_final_checkpoint=True, parameters=model.count_params(),
        original_holdout_evaluated=False, development_evaluated=False, test_evaluated=False,
        pi_tested=False, deployment_approved=False, qualification_achieved=False)
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
                    raise ValueError('Nonfinite fitting')
                write_json(out/'status.json', dict(state='running', epoch=epoch+1, metrics=logs))
                if (epoch+1) % 25 == 0:
                    print(dict(seed=seed, epoch=epoch+1, loss=logs['loss']), flush=True)

        model.fit(ds, epochs=p['epochs'], verbose=0,
            callbacks=[Progress(), keras.callbacks.CSVLogger(str(out/'epochs.csv'))])
        if int(model.optimizer.iterations) != 3600 or any(sha256(path) != h for path, h in input_sources.items()):
            raise ValueError('Schedule or source drift')
        model.save(out/'proposal.keras')
        pred = model(x, training=False).numpy()
        if pred.shape != (139, 15, 20, 6) or not np.isfinite(pred).all():
            raise ValueError('Invalid output')
        np.save(out/'predictions.npy', pred, allow_pickle=False)
        report = dict(model_sha256=sha256(out/'proposal.keras'), predictions_sha256=sha256(out/'predictions.npy'),
            provenance_sha256=sha256(out/'provenance.json'), optimizer_steps=int(model.optimizer.iterations),
            fit_loss=float(relative_loss(tf.constant(y), tf.constant(pred))),
            reference_giou_loss=float(giou_loss(tf.constant(y), tf.constant(pred))),
            relative_penalty=float(relative_penalty(tf.constant(y), tf.constant(pred))))
        write_json(out/'report.json', report)
        write_json(out/'status.json', dict(state='complete'))
        print(dict(seed=seed, **report), flush=True)
    except BaseException as exc:
        write_json(out/'status.json', dict(state='failed', error=str(exc)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=(47, 48), required=True)
    train(parser.parse_args().seed)
