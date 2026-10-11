import numpy as np

from scripts.runtime_reviewed_loss import reviewed_runtime_loss
from scripts.balloon_all_correct_protection import all_correct_loss


def test_only_reviewed_wrong_teacher_loses_preservation():
    import tensorflow as tf
    # reviewed wrong, unreviewed wrong, reviewed correct, unreviewed correct.
    target = tf.constant([[0,1,0,8,0,0,1,1], [0,1,0,8,0,0,1,0],
                          [0,1,0,0,8,0,1,1], [0,1,0,0,8,0,1,0]],tf.float32)
    logits = tf.constant([[0,2,0]]*4,tf.float32)
    actual = reviewed_runtime_loss()(target,logits).numpy()
    original = all_correct_loss()(target[:,:7],logits).numpy()
    hard = tf.nn.softmax_cross_entropy_with_logits(labels=target[:,:3],logits=logits).numpy()
    np.testing.assert_allclose(actual[0],hard[0],atol=1e-6)
    np.testing.assert_array_equal(actual[1:],original[1:])


def test_reviewed_wrong_teacher_gradient_pushes_toward_ground_truth():
    import tensorflow as tf
    target = tf.constant([[0,1,0,8,0,0,1,1]],tf.float32)
    logits = tf.Variable([[0,2.,-8.]])
    with tf.GradientTape() as tape:
        value = tf.reduce_sum(reviewed_runtime_loss()(target,logits))
    gradient = tape.gradient(value,logits).numpy()[0]
    assert gradient[1]<0 and gradient[0]>0
