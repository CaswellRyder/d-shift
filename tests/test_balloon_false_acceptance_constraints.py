import numpy as np
import pytest

pytest.importorskip('scipy.optimize')
from dtr.runtime import softmax  # noqa: E402
from scripts.balloon_head_constraints import solve  # noqa: E402
from scripts.balloon_false_acceptance_constraints import constraints  # noqa: E402


def test_bounds_cover_wrong_colors_without_relabeling_true_class():
    logits = np.array([[0., 1., .5], [2., 0., 0.], [0., 0., 5.]])
    design = np.eye(3)
    a, b, ids = constraints(design, logits, [1, 0, 0])
    assert set(ids) == {(0, -2, 1), (1, -1, 0), (1, -2, 0), (2, -1, 2)}
    delta, report = solve(a, b)
    assert report['success']
    logits[:, 1:] += design@delta.reshape(3, 2)
    p = softmax(logits)
    for row, target, _ in ids:
        assert p[row, -target] <= .8


def test_bad_labels_fail_closed():
    with pytest.raises(ValueError):
        constraints(np.ones((1, 1)), np.zeros((1, 3)), [3])
