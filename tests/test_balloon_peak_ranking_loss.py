import numpy as np
import pytest

tf = pytest.importorskip('tensorflow')
from scripts.balloon_peak_ranking_loss import local_peak_penalty  # noqa: E402


def inputs():
    target = np.zeros((1, 3, 3, 9), np.float32)
    target[..., 2:4] = 1
    target[0, 1, 1, 0] = 1
    prediction = np.zeros((1, 3, 3, 6), np.float32)
    return target, prediction


def test_center_must_outrank_reviewed_neighbor():
    target, pred = inputs()
    pred[0, 1, 2, 0] = 1
    assert float(local_peak_penalty(target, pred)) == pytest.approx(2.25)
    pred[0, 1, 1, 0] = 1.5
    assert float(local_peak_penalty(target, pred)) == 0


def test_unknown_neighbors_are_not_negative_labels():
    target, pred = inputs()
    target[0, 1, 2, 2] = 0
    pred[0, 1, 2, 0] = 100
    pred[0, 1, 1, 0] = 1
    assert float(local_peak_penalty(target, pred)) == 0


def test_adjacent_true_center_and_other_color_do_not_compete():
    target, pred = inputs()
    target[0, 1, 2, 0] = 1
    pred[0, 1, 1, 0] = 1
    pred[0, 1, 2, 0] = 4
    pred[..., 1] = 100
    assert float(local_peak_penalty(target, pred)) == 0


def test_no_centers_or_no_reviewed_neighbors_is_zero():
    target, pred = inputs()
    target[..., :2] = 0
    assert float(local_peak_penalty(target, pred)) == 0
    target[0, 1, 1, 0] = 1
    target[..., 2:4] = 0
    target[0, 1, 1, 2] = 1
    assert float(local_peak_penalty(target, pred)) == 0


def test_gradients_raise_center_lower_neighbor_and_ignore_geometry():
    target, pred = inputs()
    pred[0, 1, 2, 0] = 1
    values = tf.Variable(pred)
    with tf.GradientTape() as tape:
        loss = local_peak_penalty(target, values)
    grad = tape.gradient(loss, values).numpy()
    assert np.isfinite(grad).all()
    assert grad[0, 1, 1, 0] < 0
    assert grad[0, 1, 2, 0] > 0
    assert not grad[..., 2:].any()
    assert not grad[..., 1].any()


def test_border_center_uses_only_valid_reviewed_neighbors():
    target, pred = inputs()
    target[..., :2] = 0
    target[0, 0, 0, 0] = 1
    pred[0, 0, 1, 0] = 1
    assert float(local_peak_penalty(target, pred)) == pytest.approx(2.25)
