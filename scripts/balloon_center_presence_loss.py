"""TRAIN-only absolute confidence margin at reviewed positive centers."""


def center_presence_penalty(target, prediction):
    import tensorflow as tf
    valid = (target[..., :2] == 1.) & (target[..., 2:4] > 0)
    deficit = tf.nn.relu(tf.cast(.5, prediction.dtype) - prediction[..., :2])
    loss = tf.where(valid, tf.square(deficit), tf.zeros_like(deficit))
    return tf.math.divide_no_nan(tf.reduce_sum(loss), tf.reduce_sum(tf.cast(valid, prediction.dtype)))


def center_presence_loss(target, prediction):
    from scripts.balloon_peak_ranking_loss import peak_ranking_loss
    return peak_ranking_loss(target, prediction) + center_presence_penalty(target, prediction)
