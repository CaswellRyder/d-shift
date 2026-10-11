import numpy as np
import pytest

pytest.importorskip('scipy.special')
from scripts.balloon_head_constraints import solve  # noqa: E402
from scripts.balloon_scene_constraints import retention_constraints, decisions  # noqa: E402


def test_ambiguous_positive_is_kept_rejected_without_background_label():
    logits = np.array([[-20., 2., 2.], [0., 3., -2.]])
    design = np.eye(2)
    a, b, ids = retention_constraints(design, logits)
    delta, report = solve(a, b)
    assert report['success'] and (0, -1, 2) in ids and (0, -2, 1) in ids
    result = logits.copy()
    result[:, 1:] += design@delta.reshape(2, 2)
    np.testing.assert_array_equal(decisions(result), [0, 1])
    assert result[0, 0] == -20.  # No requirement to classify this crop as background.


def test_rejection_bounds_reject_both_colors_with_finite_margin():
    logits = np.array([[1., 0., .1]])
    a, b, _ = retention_constraints(np.ones((1, 1)), logits)
    delta, report = solve(a, b)
    assert report['success']
    after = logits.copy()
    after[:, 1:] += delta.reshape(1, 2)
    np.testing.assert_array_equal(decisions(after), [0])
