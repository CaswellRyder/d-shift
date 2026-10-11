"""Fixed paired full-feature retraining on new clutter and TRAIN scene distillation."""
import argparse
from pathlib import Path

import numpy as np

from dtr.data import sha256, write_json
from dtr.runtime import softmax
from scripts.balloon_scene_constraints import decisions
from scripts.clutter_round3_feature_data import tensors, hashes, targets, loss_function
from scripts.train_batch3_retraining import calculate


def train(arm, seed):
    out = Path(f'runs/clutter-round3-feature-{arm}-seed{seed}-20261011')
    if out.exists():
        raise FileExistsError(out)
    data = tensors(arm)
    p = data['p']
    if seed not in p['seeds']:
        raise ValueError('Undeclared seed')
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    keras.utils.set_random_seed(seed)
    model = keras.models.load_model(p['parent'], compile=False)
    if model.count_params() != p['parameters'] or model.output_shape != (None, 3):
        raise ValueError('Architecture changed')
    x, y, scene = data['x'], data['y'], data['scene_mask']
    teacher = model(x, training=False).numpy()
    probe_before = model(data['probes'], training=False).numpy()
    if not np.allclose(softmax(teacher[scene]), [r['scores'] for r in data['scene_ids']], atol=1e-5, rtol=0):
        raise ValueError('TRAIN proposal teacher identity changed')
    target = targets(y, teacher, data['preserve'], scene)
    steps = int(np.ceil(len(x)/p['batch_size']))*p['epochs']
    if steps != p['optimizer_steps']:
        raise ValueError('Schedule changed')
    sources = dict(data['sources_sha256'], **{str(__file__): sha256(__file__)})
    provenance = dict(arm=arm, seed=seed, settings=p, sources_sha256=sources,
        tensor_sha256=hashes(data), sample_identity=data['sample_identity'],
        scene_identities=data['scene_ids'], fixed_final_epoch=True, fresh_optimizer=True,
        parameters=model.count_params(), development_evaluated=False, test_evaluated=False,
        deployment_approved=False)
    out.mkdir(parents=True)
    write_json(out/'provenance.json', provenance)
    write_json(out/'status.json', dict(state='running'))
    try:
        ds = tf.data.Dataset.from_tensor_slices((x, target, data['weights']))
        options = tf.data.Options()
        options.threading.private_threadpool_size = 2
        ds = ds.shuffle(len(x), seed=seed).batch(p['batch_size']).with_options(options).prefetch(1)
        schedule = keras.optimizers.schedules.CosineDecay(p['learning_rate'], steps, alpha=p['cosine_alpha'])
        model.compile(optimizer=keras.optimizers.Adam(schedule), loss=loss_function(
            p['scene_kl_weight'], p['scene_centered_logit_weight']))

        class Progress(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if not logs or not all(np.isfinite(v) for v in logs.values()):
                    raise ValueError('Nonfinite training')
                write_json(out/'status.json', dict(state='running', epoch=epoch+1, metrics=logs))
                if (epoch+1)%10 == 0:
                    print(dict(arm=arm, seed=seed, epoch=epoch+1, metrics=logs), flush=True)

        model.fit(ds, epochs=p['epochs'], verbose=0,
            callbacks=[Progress(), keras.callbacks.CSVLogger(str(out/'epochs.csv'))])
        final = model(x, training=False).numpy()
        probe_after = model(data['probes'], training=False).numpy()
        if int(model.optimizer.iterations) != steps or not np.isfinite(final).all() or not np.isfinite(probe_after).all():
            raise ValueError('Incomplete/nonfinite training')
        report = calculate(teacher[:2074], final[:2074], y[:2074], probe_after, data['probe_labels'])
        report['scene_changed_rows'] = np.flatnonzero(decisions(teacher[scene]) != decisions(final[scene])).tolist()
        report['training_gate_pass'] &= not bool(report['scene_changed_rows'])
        model.save(out/'student.keras')
        np.savez_compressed(out/'predictions.npz', before=teacher, after=final, labels=y,
            probe_before=probe_before, probe_after=probe_after, probe_labels=data['probe_labels'])
        if any(sha256(path) != h for path, h in sources.items()):
            raise ValueError('Inputs changed during training')
        report.update(model_sha256=sha256(out/'student.keras'), predictions_sha256=sha256(out/'predictions.npz'),
            provenance_sha256=sha256(out/'provenance.json'), optimizer_steps=steps,
            scope='TRAIN-only fit/retention; no independent accuracy, Pi timing or qualification.')
        write_json(out/'report.json', report)
        write_json(out/'status.json', dict(state='complete'))
        print(dict(arm=arm, seed=seed, gate=report['training_gate_pass'],
            new_correct=sum(report['probe_correct']), lost=len(report['lost_old_rows']),
            scene_changes=len(report['scene_changed_rows'])), flush=True)
    except BaseException as exc:
        write_json(out/'status.json', dict(state='failed', error=str(exc)))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', choices=('control', 'admit'), required=True)
    p.add_argument('--seed', type=int, choices=(47, 48), required=True)
    args = p.parse_args()
    train(args.arm, args.seed)
