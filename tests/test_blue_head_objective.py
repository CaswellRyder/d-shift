import numpy as np
import pytest

from scripts.blue_head_objective import Objective,solve


def objective(negative=1.):
    rng = np.random.default_rng(47)
    return Objective(rng.normal(size=(20,4)),np.ones(20),rng.normal(size=(8,4)),
        rng.normal(size=8),negative,.0001)


def test_analytic_gradient_hessian_match_finite_difference():
    model = objective()
    x = np.array([.2,.1,-.3,.4])
    _,g,h = model.evaluate(x)
    for i in range(4):
        d = np.eye(4)[i]*1e-5
        a,ga,_ = model.evaluate(x+d)
        b,gb,_ = model.evaluate(x-d)
        assert np.isclose(g[i],(a-b)/2e-5,atol=1e-8,rtol=0)
        np.testing.assert_allclose(h[:,i],(ga-gb)/2e-5,atol=1e-8,rtol=0)
    assert np.linalg.eigvalsh(h).min() > 0


def test_control_is_exact_parent_and_background_converges_deterministically():
    d,t = solve(objective(0.))
    np.testing.assert_array_equal(d,np.zeros(4))
    assert len(t) == 1
    model = objective()
    a,ta = solve(model)
    b,tb = solve(model)
    np.testing.assert_array_equal(a,b)
    assert ta == tb and ta[-1]['gradient_inf'] <= 1e-8
    assert ta[-1]['loss'] < ta[0]['loss']
    assert all(b['loss'] <= a['loss'] for a,b in zip(ta,ta[1:]))


def test_extreme_logits_are_finite_and_invalid_inputs_fail_closed():
    m = Objective(np.eye(2),np.ones(2),np.eye(2),np.array([1000.,-1000.]),1.,.1)
    assert all(np.isfinite(v).all() for v in m.evaluate(np.zeros(2)))
    with pytest.raises(ValueError):
        Objective(np.eye(2),np.ones(2),np.eye(2),np.zeros(2),1.,0.)
    with pytest.raises(ValueError):
        m.evaluate([float('nan'),0])
    with pytest.raises(ValueError,match='did not converge'):
        solve(objective(),max_iterations=1,tolerance=1e-15)
