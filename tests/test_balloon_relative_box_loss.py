import numpy as np
import pytest

from scripts.balloon_relative_box_loss import relative_penalty, relative_loss
from scripts.balloon_dense_iou_loss import giou_loss

tf = pytest.importorskip('tensorflow')


def sample(width=20, height=20):
    target = np.zeros((1, 15, 20, 9), np.float32)
    prediction = np.zeros((1, 15, 20, 6), np.float32)
    target[0, 3, 3, 4:8] = [.25, .25, width/320, height/240]
    target[0, 3, 3, 8] = 1
    prediction[0, 3, 3, 2:] = target[0, 3, 3, 4:8]
    return target, prediction


def test_exact_geometry_zero_penalty():
    y, p = sample()
    assert float(relative_penalty(y, p)) == 0


def test_relative_center_and_size_errors_are_scale_invariant():
    small, ps = sample(20, 20)
    large, pl = sample(40, 40)
    ps[0, 3, 3, 2:4] += .1
    pl[0, 3, 3, 2:4] += .2
    ps[0, 3, 3, 4:] *= 1.5
    pl[0, 3, 3, 4:] *= 1.5
    np.testing.assert_allclose(relative_penalty(small, ps), relative_penalty(large, pl), atol=1e-7)


def test_same_pixel_center_error_costs_small_target_more():
    small, ps = sample(8, 8)
    large, pl = sample(40, 40)
    ps[0, 3, 3, 2] += .25
    pl[0, 3, 3, 2] += .25
    assert float(relative_penalty(small, ps)) > float(relative_penalty(large, pl))


def test_empty_masks_have_zero_loss_and_gradient():
    y, p = sample()
    y[..., 8] = 0
    v = tf.Variable(p)
    with tf.GradientTape() as tape:
        loss = relative_penalty(y, v)
    assert float(loss) == 0
    np.testing.assert_array_equal(tape.gradient(loss, v), np.zeros_like(p))


def test_positive_gradient_is_finite_and_only_geometry_is_updated():
    y, p = sample()
    p[0, 3, 3, 2:] += .1
    v = tf.Variable(p)
    with tf.GradientTape() as tape:
        loss = relative_penalty(y, v)
    gradient = tape.gradient(loss, v).numpy()
    assert np.isfinite(gradient).all() and np.any(gradient)
    expected_mask = np.zeros_like(gradient, bool)
    expected_mask[0, 3, 3, 2:] = True
    assert not gradient[~expected_mask].any()


def test_objective_adds_fixed_weight_without_replacing_old_loss():
    y, p = sample()
    p[0, 3, 3, 2:] += .1
    np.testing.assert_allclose(relative_loss(y, p), giou_loss(y, p)+2*relative_penalty(y, p))
