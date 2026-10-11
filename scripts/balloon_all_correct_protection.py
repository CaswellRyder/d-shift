"""Extend the frozen accepted-only protection loss to correct low-confidence positives."""
import numpy as np

from scripts.balloon_positive_protection import protected_loss


def all_correct_loss(weight=8., cap=.9, threshold=.8):
    import tensorflow as tf
    accepted_only = protected_loss(weight, cap, threshold)

    def loss(target, logits):
        labels, teacher = target[:, :3], target[:, 3:6]
        truth = tf.argmax(labels, axis=1, output_type=tf.int32)
        correct = tf.equal(tf.argmax(teacher, axis=1, output_type=tf.int32), truth)
        confidence = tf.reduce_sum(tf.nn.softmax(teacher) * labels, axis=1)
        newly_protected = (target[:, 6] > .5) & (truth > 0) & correct & (confidence < threshold)

        def log_odds(values):
            other = tf.where(labels > .5, tf.constant(-np.inf, values.dtype), values)
            return tf.reduce_sum(values * labels, axis=1) - tf.reduce_logsumexp(other, axis=1)

        desired = tf.minimum(log_odds(teacher), tf.cast(np.log(cap / (1 - cap)), logits.dtype))
        deficit = tf.nn.relu(desired - log_odds(logits))
        return accepted_only(target, logits) + weight * tf.where(newly_protected, tf.square(deficit), tf.zeros_like(deficit))

    return loss
