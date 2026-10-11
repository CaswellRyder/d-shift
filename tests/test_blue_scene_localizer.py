import numpy as np
import pytest

from scripts.blue_scene_localizer_data import negative_targets, run_path


def test_supervision_only_changes_negative_heat_masks():
    control = negative_targets(8, 'control')
    background = negative_targets(8, 'background')
    assert control.shape == background.shape == (8, 15, 20, 9)
    assert not np.any(control)
    assert np.all(background[..., 2:4] == 1)
    assert not np.any(background[..., :2])
    assert not np.any(background[..., 4:])
    assert int(background.sum()) == 4800


def test_reject_undeclared_experiment():
    with pytest.raises(ValueError):
        negative_targets(8, 'unknown')
    with pytest.raises(ValueError):
        negative_targets(0, 'control')
    with pytest.raises(ValueError):
        run_path('background', 49)


def test_negative_loss_no_geometry_gradient():
    tf = pytest.importorskip('tensorflow')
    from scripts.balloon_dense_iou_loss import giou_loss
    pred = tf.Variable(np.zeros((1, 15, 20, 6), np.float32))
    with tf.GradientTape() as tape:
        loss = giou_loss(tf.constant(negative_targets(1, 'background')), pred)
    grad = tape.gradient(loss, pred).numpy()
    assert float(loss) > 0
    assert np.all(grad[..., :2] > 0)
    assert not np.any(grad[..., 2:])
    with tf.GradientTape() as tape:
        loss = giou_loss(tf.constant(negative_targets(1, 'control')), pred)
    assert float(loss) == 0
    assert not np.any(tape.gradient(loss, pred).numpy())
