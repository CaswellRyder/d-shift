import numpy as np
import pytest

pytest.importorskip('scipy.optimize')
from scripts.balloon_minimum_drift import quadratic, solve  # noqa: E402


def test_convex_solution_and_independent_optimality_certificate():
    result, report = solve(np.array([[1., 1.]]), np.array([1.]), np.diag([1., 4.]), np.array([1., 0.]))
    np.testing.assert_allclose(result, [.8, .2], atol=1e-6)
    assert report['success'] and report['stationarity'] < 1e-5
    assert report['objective'] == pytest.approx(.4)


def test_logit_drift_objective_uses_interleaved_class_columns():
    x = np.array([[1., 2.], [-1., .5]])
    delta = np.array([[.3, -.2], [1., .4]])
    h = quadratic(x)
    assert delta.ravel()@h@delta.ravel() == pytest.approx(np.sum((x@delta)**2)/len(x)+.001*np.sum(delta**2))


def test_infeasible_initial_solution_fails_closed():
    with pytest.raises(ValueError):
        solve(np.eye(2), np.ones(2), np.eye(2), np.zeros(2))
