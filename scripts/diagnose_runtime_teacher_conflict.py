"""Check whether inherited soft targets oppose reviewed TRAIN-positive labels.

This optimizes independent three-logit vectors, not network weights. It is a
loss diagnostic, never a model checkpoint or accuracy result.
"""
from pathlib import Path

import numpy as np

from dtr.data import read_json,sha256,write_json
from dtr.runtime import softmax
from scripts.audit_runtime_positives import OUTPUT as TRIAL
from scripts.audit_runtime_positive_coverage import QUEUE
from scripts.balloon_all_correct_protection import all_correct_loss

OUTPUT = Path('research/balloon-red-blue-20261008/runtime-teacher-conflict-20261011.json')


def diagnose():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    trial = read_json(TRIAL)
    queue = read_json(QUEUE/'review.json')
    rows = [r for r in queue['samples'] if not r['in_training']]
    with np.load(Path(trial['trials']['admit47']['path'])/'predictions.npz') as a:
        teacher,labels = a['runtime_before'],a['runtime_labels']
    if len(rows)!=139 or not np.array_equal(labels,[1 if r['label']=='red_balloon' else 2 for r in rows]):
        raise ValueError('Runtime source order changed')
    for r in trial['trials'].values():
        if sha256(Path(r['path'])/'predictions.npz')!=r['predictions_sha256']:
            raise ValueError('Prediction receipt changed')
    indices = np.flatnonzero(teacher.argmax(1)!=labels)
    import tensorflow as tf
    tf.config.set_visible_devices([],'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    targets = tf.constant(np.column_stack((np.eye(3)[labels[indices]],teacher[indices],np.ones(len(indices)))).astype(np.float32))
    logits = tf.Variable(teacher[indices])
    optimizer = tf.keras.optimizers.Adam(.03)
    loss = all_correct_loss(8.,.9,.8)

    @tf.function
    def step():
        with tf.GradientTape() as tape:
            value = tf.reduce_sum(loss(targets,logits))
        gradient = tape.gradient(value,logits)
        optimizer.apply_gradients([(gradient,logits)])
        return value

    for _ in range(2000):
        step()
    with tf.GradientTape() as tape:
        value = tf.reduce_sum(loss(targets,logits))
    gradient = tape.gradient(value,logits).numpy()
    probabilities = softmax(logits.numpy())
    results = []
    for j,i in enumerate(indices):
        examples = {}
        for name,report in trial['trials'].items():
            with np.load(Path(report['path'])/'predictions.npz') as a:
                examples[name] = softmax(a['runtime_after'])[i].tolist()
        results.append(dict(queue_id=rows[i]['id'],source=rows[i]['source'],label=rows[i]['label'],
            teacher_probability=softmax(teacher)[i].tolist(),network_probabilities=examples,
            independent_logit_optimum=probabilities[j].tolist(),max_abs_gradient=float(np.max(np.abs(gradient[j]))),
            optimum_passes_threshold=bool(probabilities[j].argmax()==labels[i] and probabilities[j,labels[i]]>=.8)))
    write_json(OUTPUT,dict(results=results,trial_sha256=sha256(TRIAL),queue_sha256=sha256(QUEUE/'review.json'),
        script_sha256=sha256(__file__),loss_helpers_sha256={p:sha256(p) for p in (
            'scripts/balloon_all_correct_protection.py','scripts/balloon_positive_protection.py','scripts/train_balloon_reviewed_clutter.py')},
        steps=2000,optimizer='Adam0.03',scope='Independent-logit TRAIN loss diagnostic. No network retraining, deployment or accuracy claim.',
        test_evaluated=False,deployment_approved=False,qualification_achieved=False))
    print(results)


if __name__=='__main__':
    diagnose()
