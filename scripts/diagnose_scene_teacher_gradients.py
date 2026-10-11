"""Measure which TRAIN objectives oppose the10reviewed corrections, without fitting."""
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from scripts.scene_teacher_supervision import tensors, targets, loss_function, PROTOCOL
from scripts.audit_scene_teacher_supervision import OUTPUT as AUDIT
from scripts.balloon_gradient_summary import summarize

OUTPUT = Path('research/balloon-red-blue-20261008/scene-teacher-gradients-20261011.json')
ARCHIVE = Path('runs/scene-teacher-gradients-20261011.npz')


def diagnose():
    if OUTPUT.exists() or ARCHIVE.exists():
        raise FileExistsError('Preserve objective diagnostic')
    audit, data = read_json(AUDIT), tensors()
    if (audit['script_sha256'] != sha256('scripts/audit_scene_teacher_supervision.py')
            or audit['protocol_sha256'] != sha256(PROTOCOL)):
        raise ValueError('Training audit changed')
    ids = np.arange(2511)
    groups = dict(old_crops=ids[:2074], new_negatives=ids[2074:2126],
        retained_scenes=ids[(ids >= 2126) & (data['correction_labels'] < 0)],
        reviewed_corrections=ids[data['correction_labels'] >= 0])
    if [len(x) for x in groups.values()] != [2074, 52, 375, 10]:
        raise ValueError('Disjoint TRAIN objective groups changed')
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    parent = keras.models.load_model(data['p']['parent'], compile=False)
    target = targets(data, parent(data['x'], training=False).numpy())
    objective = loss_function()
    paths = {'parent': Path(data['p']['parent'])}
    paths.update({f'reviewed{s}': Path(audit['results'][f'reviewed{s}']['run'])/'student.keras' for s in (47, 48)})
    results, archive = {}, {}
    for name, path in paths.items():
        digest = data['p']['parent_sha256'] if name == 'parent' else audit['results'][name]['model_sha256']
        if sha256(path) != digest:
            raise ValueError('Checkpoint changed')
        model = keras.models.load_model(path, compile=False)
        weights = [v.numpy().copy() for v in model.weights]

        @tf.function(reduce_retracing=True)
        def gradient(x, y, w):
            with tf.GradientTape() as tape:
                values = objective(y, model(x, training=False))*w
                total = tf.reduce_sum(values)
            grads = tape.gradient(total, model.trainable_weights)
            return values, tf.concat([tf.reshape(tf.zeros_like(v) if g is None else g, [-1])
                for g, v in zip(grads, model.trainable_weights, strict=True)], axis=0)

        gradients, losses = {}, {}
        for group, indices in groups.items():
            accumulated, values = np.zeros(7763), []
            for start in range(0, len(indices), 64):
                selected = indices[start:start+64]
                loss, grad = gradient(data['x'][selected], target[selected], data['weights'][selected])
                accumulated += grad.numpy()
                values.extend(loss.numpy().tolist())
            gradients[group] = accumulated/len(indices)
            losses[group] = dict(rows=len(indices), mean_loss=float(np.mean(values)),
                                 contribution=float(np.sum(values)/2511))
            archive[f'{name}/{group}/gradient'] = gradients[group]
            archive[f'{name}/{group}/losses'] = np.asarray(values)
        if not all(np.array_equal(a, b.numpy()) for a, b in zip(weights, model.weights, strict=True)):
            raise ValueError('Diagnostic changed weights')
        summary = summarize(gradients, {k: len(v) for k, v in groups.items()})
        results[name] = dict(model=str(path), model_sha256=digest, losses=losses, gradients=summary)
        print(dict(name=name, losses=losses, gradients=summary), flush=True)
    np.savez_compressed(ARCHIVE, **archive)
    # Independently regenerate reductions from the saved vectors, not printed output.
    with np.load(ARCHIVE, allow_pickle=False) as saved:
        for name, row in results.items():
            calculated = summarize({g: saved[f'{name}/{g}/gradient'] for g in groups},
                                   {g: len(v) for g, v in groups.items()})
            if calculated != row['gradients']:
                raise ValueError('Archived gradients do not reproduce')
            for g, indices in groups.items():
                values = saved[f'{name}/{g}/losses']
                if values.shape != (len(indices),) or not np.isfinite(values).all():
                    raise ValueError('Invalid archived objectives')
    write_json(OUTPUT, dict(results=results, audit_sha256=sha256(AUDIT),
        sources_sha256={str(p): sha256(p) for p in (__file__, PROTOCOL, 'scripts/scene_teacher_supervision.py',
                                                   'scripts/balloon_gradient_summary.py')},
        archive=dict(path=str(ARCHIVE), sha256=sha256(ARCHIVE)),
        scope='Frozen TRAIN loss/gradient diagnosis only. Negative cosine is local opposition, not a proof of insufficient model capacity or of an impossible fit.',
        weights_changed=False, training_performed=False, development_evaluated=False,
        test_evaluated=False, pi_tested=False, deployment_approved=False, qualification_achieved=False))


if __name__ == '__main__':
    diagnose()
