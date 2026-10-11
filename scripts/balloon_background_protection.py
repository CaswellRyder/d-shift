"""Training-only protection of correct old background argmax decisions."""
from scripts.balloon_all_correct_protection import all_correct_loss


def background_penalty(target, logits, weight=8., cap=.9):
    import tensorflow as tf
    import math
    if weight <= 0 or not .5 < cap < 1:
        raise ValueError('Invalid background protection settings')
    labels, teacher = target[:, :3], target[:, 3:6]
    eligible = ((target[:, 7] > .5) & (tf.argmax(labels, axis=1) == 0)
                & (tf.argmax(teacher, axis=1) == 0))
    # Background acceptance means argmax, not probability>=0.8. Preserve its
    # margin against the strongest foreground instead of against their sum.
    reference = teacher[:, 0]-tf.reduce_max(teacher[:, 1:], axis=1)
    desired = tf.minimum(reference, tf.cast(math.log(cap/(1-cap)), logits.dtype))
    margin = logits[:, 0]-tf.reduce_max(logits[:, 1:], axis=1)
    deficit = tf.nn.relu(desired-margin)
    return weight*tf.where(eligible, tf.square(deficit), tf.zeros_like(deficit))


def loss_function(weight=8., cap=.9, threshold=.8):
    base = all_correct_loss(weight, cap, threshold)

    def loss(target, logits):
        return base(target, logits)+background_penalty(target, logits, weight, cap)

    return loss
