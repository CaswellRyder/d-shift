"""New scene labels affect supervision only, not matched input tensors."""
import numpy as np
import pytest

from scripts.public_scene_localizer_data import supervise, run_path


def test_control_masks_every_new_loss_but_keeps_target_values():
    truth = [dict(label='blue_balloon', box=[30, 30, 60, 60]),
             dict(label='red_balloon', box=[130, 130, 160, 160])]
    a, b = supervise(truth, 'control'), supervise(truth, 'scenes')
    assert not a[..., [2, 3, 8]].any()
    np.testing.assert_array_equal(a[..., [0, 1, 4, 5, 6, 7]], b[..., [0, 1, 4, 5, 6, 7]])
    assert b[..., 8].sum() == 2
    assert b[..., 2:4].sum() > 2


def test_overlapping_centers_fail_closed():
    with pytest.raises(ValueError, match='silently drop'):
        supervise([dict(label='blue_balloon', box=[1, 1, 10, 10]),
                   dict(label='red_balloon', box=[2, 2, 12, 12])], 'scenes')


def test_undeclared_arm():
    with pytest.raises(ValueError):
        supervise([], 'test')


@pytest.mark.parametrize('arm,seed', [('test', 47), ('scenes', 99)])
def test_undeclared_run(arm, seed):
    with pytest.raises(ValueError):
        run_path(arm, seed)
