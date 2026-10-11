"""Training-only size-relative localization penalty; inference graph is unchanged."""


def relative_penalty(target, prediction):
    import tensorflow as tf
    valid = target[..., 8] > 0
    truth = tf.boolean_mask(target[..., 4:8], valid)
    actual = tf.boolean_mask(prediction[..., 2:6], valid)
    size = tf.constant([320., 240.])
    truth_pixels = truth[..., 2:] * size
    center_error = (actual[..., :2] - truth[..., :2]) * 16 / tf.maximum(truth_pixels, 4.)
    size_error = tf.math.log(tf.maximum(actual[..., 2:] * size, 1.) / tf.maximum(truth_pixels, 1.))
    residual = tf.abs(tf.concat([center_error, size_error], axis=-1))
    smooth_l1 = tf.where(residual < 1., .5 * residual**2, residual - .5)
    return tf.reduce_sum(smooth_l1) / tf.maximum(tf.cast(tf.size(smooth_l1), tf.float32), 1.)


def relative_loss(target, prediction):
    from scripts.balloon_dense_iou_loss import giou_loss
    return giou_loss(target, prediction) + 2 * relative_penalty(target, prediction)
