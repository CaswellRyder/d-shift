import numpy as np
import pytest

from scripts.blue_soft_shape_fusion import adjust, fit, odds


def test_no_correction_preserves_parent_exactly():
    p = np.array([[.2, .1, .7], [.1, .85, .05], [0., 0., 1.]])
    model = dict(mean=np.zeros(2), scale=np.ones(2), coefficients=np.zeros(2), bias=np.array(0.))
    assert np.array_equal(adjust(p, np.ones((3, 2)), model), p)


def test_fusion_preserves_red_and_mass_and_can_rescue_or_reject_blue():
    p = np.array([[.2, .1, .7], [.1, .85, .05], [.7, .1, .2]])
    model = dict(mean=np.zeros(1), scale=np.ones(1), coefficients=np.ones(1), bias=np.array(0.))
    actual = adjust(p, np.array([[-5.], [5.], [5.]]), model)
    assert np.array_equal(actual[:, 1], p[:, 1]) and np.array_equal(actual[1], p[1])
    assert actual[0, 2] < p[0, 2] and actual[2, 2] > p[2, 2]
    assert np.allclose(actual.sum(1), 1.) and np.all(actual >= 0)


def test_fit_does_not_use_red_features():
    x, labels = np.array([[0.], [1.], [10.]]), np.array([0, 2, 1])
    p = np.array([[.1, .1, .8], [.1, .1, .8], [.1, .8, .1]])
    a = fit(x, labels, p)
    x[2] = 1e8
    b = fit(x, labels, p)
    assert all(np.array_equal(a[k], b[k]) for k in a)


def test_invalid_probabilities_fail():
    with pytest.raises(ValueError, match='probabilities'):
        odds([[.9, .9, .9]])
