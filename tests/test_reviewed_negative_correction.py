import numpy as np
import pytest

from scripts.reviewed_negative_correction_data import correct_scene_targets, run_path


def test_only_explicit_scene_targets_change_without_mutating_inputs():
    y = np.array([1, -1, -1, 0])
    scene = np.array([False, True, True, False])
    preserve = np.array([True, False, False, False])
    weights = np.array([1., 1., 1., 4.])
    a, b, c, d = correct_scene_targets(y, scene, preserve, weights, [2])
    np.testing.assert_array_equal(a, [1, -1, 0, 0])
    np.testing.assert_array_equal(b, [False, True, False, False])
    np.testing.assert_array_equal(c, preserve)
    np.testing.assert_array_equal(d, [1., 1., 4., 4.])
    assert y[2] == -1 and scene[2] and weights[2] == 1


@pytest.mark.parametrize('positions', [[], [0], [1, 1], [-1], [2]])
def test_non_scene_unknown_and_duplicate_rows_fail(positions):
    with pytest.raises(ValueError):
        correct_scene_targets(np.array([0, -1]), np.array([False, True]),
                              np.array([False, False]), np.ones(2), positions)


def test_undeclared_run_fails():
    with pytest.raises(ValueError):
        run_path('corrected', 49)
