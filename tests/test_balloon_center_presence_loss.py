import numpy as np
import pytest

tf = pytest.importorskip('tensorflow')
from scripts.balloon_center_presence_loss import center_presence_penalty  # noqa: E402


def test_only_annotated_supervised_centers_receive_presence_loss():
    t, p = np.zeros((1, 3, 3, 9), np.float32), np.zeros((1, 3, 3, 6), np.float32)
    assert float(center_presence_penalty(t, p)) == 0
    t[0, 1, 1, 1] = 1
    assert float(center_presence_penalty(t, p)) == 0
    t[0, 1, 1, 3] = 1
    assert float(center_presence_penalty(t, p)) == pytest.approx(.25)
    p[0, 1, 1, 1] = .5
    assert float(center_presence_penalty(t, p)) == 0


def test_adjacent_centers_are_both_supported_and_unknown_cells_ignored():
    t, p = np.zeros((1, 3, 3, 9), np.float32), np.full((1, 3, 3, 6), -100, np.float32)
    t[0, 1, 1:3, 0] = 1
    t[0, 1, 1:3, 2] = 1
    p[0, 1, 1, 0], p[0, 1, 2, 0] = 0, -.5
    assert float(center_presence_penalty(t, p)) == pytest.approx(.625)


def test_gradient_raises_only_true_center_and_is_finite():
    t = np.zeros((1, 3, 3, 9), np.float32)
    t[0, 1, 1, 0], t[0, 1, 1, 2] = 1, 1
    p = tf.Variable(np.zeros((1, 3, 3, 6), np.float32))
    with tf.GradientTape() as tape:
        loss = center_presence_penalty(t, p)
    g = tape.gradient(loss, p).numpy()
    assert np.isfinite(g).all() and g[0, 1, 1, 0] == -1
    assert np.count_nonzero(g) == 1
