import numpy as np
import pytest

from scripts.balloon_background_protection import background_penalty, loss_function
from scripts.balloon_all_correct_protection import all_correct_loss

tf = pytest.importorskip('tensorflow')


def targets():
    # correct-old background, new background, wrong-parent background, positive.
    return tf.constant([[1.,0,0, 2.,1.,0., 1.,1.], [1.,0,0, 2.,1.,0., 0.,0.],
                        [1.,0,0, 0.,2.,1., 1.,1.], [0.,1.,0, 0.,2.,1., 1.,1.]])


def test_only_correct_old_background_receives_extra_penalty():
    t = targets()
    logits = tf.constant([[0.,1.,0.]]*4)
    np.testing.assert_allclose(background_penalty(t, logits).numpy(), [32.,0,0,0])
    expected = all_correct_loss()(t, logits)+background_penalty(t, logits)
    np.testing.assert_allclose(loss_function()(t, logits).numpy(), expected.numpy())


def test_unchanged_or_improved_margin_has_zero_cost():
    t = targets()
    np.testing.assert_array_equal(background_penalty(t, t[:,3:6]), np.zeros(4))
    np.testing.assert_array_equal(background_penalty(t, tf.constant([[3.,1.,0.]]*4)), np.zeros(4))


def test_argmax_margin_protects_low_probability_background_and_caps_high_margin():
    t = tf.constant([[1.,0,0, .1,0.,0., 1.,1.], [1.,0,0, 100.,0.,0., 1.,1.]])
    penalties = background_penalty(t, tf.zeros((2,3))).numpy()
    np.testing.assert_allclose(penalties, [8*.1**2, 8*np.log(9)**2], rtol=1e-5)


def test_gradient_restores_background_without_touching_ineligible_rows():
    z = tf.Variable([[0.,1.,0.]]*4)
    with tf.GradientTape() as tape:
        value = tf.reduce_sum(background_penalty(targets(), z))
    gradient = tape.gradient(value, z).numpy()
    assert gradient[0,0] < 0 and gradient[0,1] > 0
    np.testing.assert_array_equal(gradient[1:], np.zeros((3,3)))


@pytest.mark.parametrize('weight,cap', [(0,.9), (8,.5), (8,1.)])
def test_invalid_settings_fail(weight, cap):
    with pytest.raises(ValueError):
        background_penalty(targets(), tf.zeros((4,3)), weight, cap)
