"""Training-only one-sided protection of previously correct positive decisions."""
import numpy as np

from dtr.runtime import softmax
from scripts.train_balloon_reviewed_clutter import CLASSES, loss_function as previous_loss


def protected_loss(weight=8., cap=.9, threshold=.8):
    import tensorflow as tf
    if weight <= 0 or not .5 < threshold < cap < 1:
        raise ValueError('Invalid protection settings')
    base_loss = previous_loss()

    def loss(target, logits):
        labels, teacher = target[:, :3], target[:, 3:6]
        ordinary = target[:, 6] > .5
        truth = tf.argmax(labels, axis=1, output_type=tf.int32)
        parent_scores = tf.nn.softmax(teacher)
        parent_correct = tf.equal(tf.argmax(teacher, axis=1, output_type=tf.int32), truth)
        parent_accepted = tf.reduce_sum(parent_scores * labels, axis=1) >= threshold
        protected = ordinary & (truth > 0) & parent_correct & parent_accepted

        def log_odds(values):
            other = tf.where(labels > .5, tf.constant(-np.inf, values.dtype), values)
            return tf.reduce_sum(values * labels, axis=1) - tf.reduce_logsumexp(other, axis=1)

        desired = tf.minimum(log_odds(teacher), tf.cast(np.log(cap / (1 - cap)), logits.dtype))
        shortfall = tf.nn.relu(desired - log_odds(logits))
        penalty = tf.where(protected, tf.square(shortfall), tf.zeros_like(shortfall))
        return base_loss(target, logits) + weight * penalty

    return loss


def retention(before, after, labels, ordinary, threshold=.8):
    labels, ordinary = np.asarray(labels), np.asarray(ordinary, bool)
    before, after = np.asarray(before), np.asarray(after)
    if (before.shape != after.shape or before.shape != (len(labels), 3)
            or ordinary.shape != labels.shape or not np.isfinite(before).all() or not np.isfinite(after).all()):
        raise ValueError('Invalid prediction receipt')
    bp, ap = softmax(before), softmax(after)
    result = {}
    for color in (1, 2):
        mask = ordinary & (labels == color)
        parent_accept = mask & (before.argmax(1) == color) & (bp[:, color] >= threshold)
        now_accept = (after.argmax(1) == color) & (ap[:, color] >= threshold)
        lost = parent_accept & ~now_accept
        result[CLASSES[color]] = dict(total=int(mask.sum()),
            parent_argmax_correct=int((mask & (before.argmax(1) == color)).sum()),
            current_argmax_correct=int((mask & (after.argmax(1) == color)).sum()),
            parent_accepted=int(parent_accept.sum()), retained=int((parent_accept & now_accept).sum()),
            lost_indices=np.flatnonzero(lost).tolist(),
            retained_fraction=float((parent_accept & now_accept).sum() / parent_accept.sum()) if parent_accept.any() else None)
    return result


def feasible(report, protocol):
    return (set(report['retention']) == {'red_balloon', 'blue_balloon'}
            and all(r['retained_fraction'] is not None
                and r['retained_fraction'] >= protocol['minimum_parent_acceptance_retained']
                and r['current_argmax_correct'] >= r['parent_argmax_correct']
                for r in report['retention'].values())
            and report['reviewed_background_correct'] >= protocol['minimum_reviewed_background_correct'])
