"""TRAIN-only gradient attribution of the frozen window1 objective; no fitting."""
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from dtr.runtime import softmax
from scripts.window1_trial_data import tensors
from scripts.clutter_round3_feature_data import hashes, targets, loss_function
from scripts.audit_window1_runtime import OUTPUT as AUDIT

OUTPUT = Path('research/balloon-red-blue-20261008/window1-objective-20261011.json')


def gradient_stats(groups, margin_grads):
    values = {k: np.asarray(v, np.float64) for k, v in groups.items()}
    if not values or len({v.shape for v in values.values()}) != 1:
        raise ValueError('Inconsistent gradient dimensions')
    total = sum(values.values())
    new = values['new_positive']
    result = {}
    for name, grad in dict(values, total=total).items():
        if grad.ndim != 1 or not np.isfinite(grad).all():
            raise ValueError('Invalid gradient')
        denom = np.linalg.norm(grad)*np.linalg.norm(new)
        result[name] = dict(norm=float(np.linalg.norm(grad)),
            dot_new=float(grad @ new), cosine_new=float(grad @ new/denom) if denom else None,
            negative_gradient_margin_derivatives=[float(-np.asarray(m) @ grad) for m in margin_grads])
    return result


def diagnose():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    data = tensors('hard')
    audit = read_json(AUDIT)
    if any(sha256(p) != h for p, h in audit['inputs_sha256'].items()):
        raise ValueError('Frozen training audit changed')
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    parent = keras.models.load_model(data['p']['parent'], compile=False)
    teacher = parent(data['x'], training=False).numpy()
    target = targets(data['y'], teacher, data['preserve'], data['scene_mask'])
    loss = loss_function(data['p']['scene_kl_weight'], data['p']['scene_centered_logit_weight'])
    indexes = dict(old_supervised=np.arange(2074), new_negative=np.arange(2074, 2126),
                   new_positive=np.arange(2126, 2162), scene_distillation=np.arange(2162, 2547))
    n = len(data['x'])
    sources = dict(data['sources_sha256'])
    sources[str(AUDIT)] = sha256(AUDIT)
    results = {}

    def flat(grads, variables):
        return np.concatenate([(np.zeros(v.shape) if g is None else g.numpy()).ravel()
                               for g, v in zip(grads, variables, strict=True)])

    for name in ('parent', 'hard47', 'hard48'):
        if name == 'parent':
            model = parent
        else:
            row = audit['results'][name]
            root = Path(row['run'])
            prov = read_json(root/'provenance.json')
            if (prov['tensor_sha256'] != hashes(data)
                    or any(sha256(root/f) != h for f, h in row['artifacts_sha256'].items())):
                raise ValueError('Run provenance changed')
            model = keras.models.load_model(root/'student.keras', compile=False)
            sources[str(root/'student.keras')] = sha256(root/'student.keras')
        variables = model.trainable_variables
        gradients, losses = {}, {}
        for group, ids in indexes.items():
            summed = np.zeros(sum(int(np.prod(v.shape)) for v in variables), np.float64)
            total_loss = 0.
            for offset in range(0, len(ids), 64):
                rows = ids[offset:offset+64]
                with tf.GradientTape() as tape:
                    logits = model(data['x'][rows], training=False)
                    value = tf.reduce_sum(loss(tf.constant(target[rows]), logits)*data['weights'][rows])/n
                summed += flat(tape.gradient(value, variables), variables)
                total_loss += float(value.numpy())
            gradients[group], losses[group] = summed, total_loss
        margin_gradients = []
        for pixels in data['batch_pixels'][-2:]:
            with tf.GradientTape() as tape:
                logits = model(pixels[None], training=False)[0]
                margin = logits[2]-tf.reduce_logsumexp(logits[:2])
            margin_gradients.append(flat(tape.gradient(margin, variables), variables))
        scores = softmax(model(data['batch_pixels'][-2:], training=False).numpy())
        results[name] = dict(group_weighted_loss=losses, objective=sum(losses.values()),
            gradients=gradient_stats(gradients, margin_gradients), runtime_probabilities=scores.tolist())
        print(dict(name=name, objective=sum(losses.values()), groups=results[name]['gradients']), flush=True)
    sources[str(__file__)] = sha256(__file__)
    if any(sha256(p) != h for p, h in sources.items()):
        raise ValueError('Diagnostic sources changed')
    write_json(OUTPUT, dict(results=results, inputs_sha256=sources,
        groups={k:dict(rows=len(v), weight_sum=float(data['weights'][v].sum())) for k, v in indexes.items()},
        scope='Full-batch gradient attribution at frozen TRAIN checkpoints, normalized by total row count. Negative-gradient margin derivatives are local SGD directions, not Adam updates, accuracy, or proof that reweighting will generalize. No fitting, new labels, threshold changes or model selection.',
        training_performed=False, development_evaluated=False, test_evaluated=False,
        pi_tested=False, deployment_approved=False, qualification_achieved=False))


if __name__ == '__main__':
    diagnose()
