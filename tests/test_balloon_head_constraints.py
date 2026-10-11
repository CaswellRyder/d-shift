import numpy as np
import pytest

softmax = pytest.importorskip('scipy.special').softmax
from scripts.balloon_head_constraints import constraints, solve  # noqa: E402


def test_protected_positive_bounds_jointly_guarantee_threshold():
    logits = np.array([[0., 2., -3.]])
    a, b, ids = constraints(np.ones((1, 1)), logits, [1], [True], [False])
    delta, report = solve(a, b)
    assert report['success'] and ids == [(0, 1, 0), (0, 1, 2)]
    after = logits.copy()
    after[:, 1:] += delta.reshape(1, 2)
    assert softmax(after, axis=1)[0, 1] >= .8


def test_new_positive_learned_without_changing_background_logits():
    x = np.array([[1., 0.], [0., 1.]])
    logits = np.array([[2., 0., 0.], [1., 0., 0.]])
    a, b, _ = constraints(x, logits, [0, 2], [True, False], [False, True])
    delta, report = solve(a, b)
    assert report['success']
    after = logits.copy()
    after[:, 1:] += x@delta.reshape(2, 2)
    assert after[0].argmax() == 0 and softmax(after, axis=1)[1, 2] >= .81-1e-8
    np.testing.assert_array_equal(after[:, 0], logits[:, 0])


def test_conflicting_identical_features_report_infeasible_not_success():
    x, logits = np.ones((2, 1)), np.array([[2., 0., 0.], [2., 0., 0.]])
    a, b, _ = constraints(x, logits, [0, 2], [True, False], [False, True])
    delta, report = solve(a, b)
    assert delta is None and report['status'] == 2 and not report['success']
