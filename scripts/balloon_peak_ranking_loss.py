"""TRAIN-only center ordering loss for the unchanged three-by-three peak decoder.

Only explicitly supervised same-color non-center neighbors compete with a true
center. Unknown cells and adjacent true object centers never become negatives.
The term changes fitting, not the deployed inference graph or decoder thresholds.
"""


def local_peak_penalty(target, prediction, margin=.5):
    import tensorflow as tf
    heat, supervised = target[..., :2], target[..., 2:4] > 0
    centers = supervised & (heat == 1.)
    negatives = supervised & (heat < 1.)
    logits = prediction[..., :2]
    masked = tf.where(negatives, logits, tf.cast(-1e9, logits.dtype))
    neighbor = tf.nn.max_pool2d(masked, ksize=3, strides=1, padding='SAME')
    has_neighbor = tf.nn.max_pool2d(tf.cast(negatives, logits.dtype), 3, 1, 'SAME') > 0
    valid = centers & has_neighbor
    deficit = tf.nn.relu(tf.cast(margin, logits.dtype) + neighbor - logits)
    penalties = tf.where(valid, tf.square(deficit), tf.zeros_like(deficit))
    return tf.math.divide_no_nan(tf.reduce_sum(penalties), tf.reduce_sum(tf.cast(valid, logits.dtype)))


def peak_ranking_loss(target, prediction):
    from scripts.balloon_relative_box_loss import relative_loss
    return relative_loss(target, prediction) + local_peak_penalty(target, prediction)
