import numpy as np
import pytest

from scripts.blue_prototype_rejection import adjust, correction, fit


def test_positive_exclusion_and_negative_center_correction():
    features = np.array([[0.,0.], [1.,1.], [10.,10.]])
    labels = np.array([1,2,0])
    bank = fit(features, labels, features[2:], np.array([[0.,-5.,5.]]))
    delta = correction(features, bank)
    assert np.array_equal(delta[:2], [0.,0.])
    assert delta[2] == 13.
    assert correction(np.array([[100.,100.]]), bank)[0] == 0


def test_zero_distance_negative_conflict_cannot_reject_positive():
    features = np.array([[1.,2.], [1.,2.]])
    bank = fit(features, [1,0], features[1:], [[0.,0.,8.]])
    assert np.array_equal(correction(features, bank), [0.,0.])


def test_red_probability_is_exact_and_blue_never_increases():
    rng = np.random.default_rng(47)
    scores = rng.dirichlet([1.,1.,1.], 1000)
    delta = rng.uniform(0,50,1000)
    values = adjust(scores, delta)
    assert np.array_equal(values[:,1], scores[:,1])
    assert np.all(values[:,2] <= scores[:,2])
    np.testing.assert_allclose(values.sum(axis=1), scores.sum(axis=1), atol=1e-15)
    assert np.array_equal(adjust(scores,np.zeros(1000)),scores)


def test_probability_edges_remain_finite():
    scores = np.array([[1.,0.,0.], [0.,1.,0.], [0.,0.,1.]])
    actual = adjust(scores, [10000.,10000.,10000.])
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual.sum(axis=1), 1.)


@pytest.mark.parametrize('delta', [[-1.], [float('nan')], [float('inf')]])
def test_invalid_correction_rejected(delta):
    with pytest.raises(ValueError):
        adjust([[.1,.1,.8]],delta)
