"""One-sided scene decision retention instead of exact teacher confidence fitting."""
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256
from scripts.scene_teacher_supervision import tensors as reviewed_tensors, loss_function as reviewed_loss

PROTOCOL = Path('configs/scene-decision-retention-20261011.json')


def tensors():
    data = reviewed_tensors()
    for p in (PROTOCOL, __file__, 'research/balloon-red-blue-20261008/scene-teacher-gradients-20261011.json'):
        data['sources_sha256'][str(p)] = sha256(p)
    return data


def log_odds(logits):
    import tensorflow as tf
    return tf.stack([logits[:, c]-tf.reduce_logsumexp(tf.gather(logits, [k for k in range(3) if k != c], axis=1), axis=1)
                     for c in range(3)], axis=1)


def decision_loss(teacher, logits):
    import tensorflow as tf
    p = read_json(PROTOCOL)
    parent, current = log_odds(teacher), log_odds(logits)
    label = tf.argmax(teacher, axis=1, output_type=tf.int32)
    confidence = tf.reduce_max(tf.nn.softmax(teacher), axis=1)
    accepted = (label > 0) & (confidence >= p['threshold'])
    selected = tf.one_hot(label, 3, dtype=logits.dtype)
    floor = tf.minimum(tf.reduce_sum(parent*selected, axis=1), np.log(p['accepted_cap']/(1-p['accepted_cap'])))
    accepted_loss = tf.square(tf.nn.relu(floor-tf.reduce_sum(current*selected, axis=1)))
    ceiling = tf.maximum(parent[:, 1:], np.log(p['rejected_floor']/(1-p['rejected_floor'])))
    rejected_loss = tf.reduce_sum(tf.square(tf.nn.relu(current[:, 1:]-ceiling)), axis=1)
    return p['weight']*tf.where(accepted, accepted_loss, rejected_loss)


def loss_function():
    import tensorflow as tf
    baseline = reviewed_loss()

    def loss(target, logits):
        retained = (target[:, 7] > .5) & (target[:, 11] < .5)
        return tf.where(retained, decision_loss(target[:, 3:6], logits), baseline(target, logits))

    return loss
