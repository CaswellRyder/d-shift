import numpy as np
import pytest

from scripts.balloon_dense_context_lift import build, compose_kernel, cost


def test_kernel_composition_and_invalid_contracts():
    d = np.ones((3, 3, 32, 1), np.float32)
    p = np.arange(1024, dtype=np.float32).reshape(1, 1, 32, 32)
    actual = compose_kernel(d, p)
    assert actual.shape == (3, 3, 32, 32)
    np.testing.assert_array_equal(actual[1, 2], p[0, 0])
    with pytest.raises(ValueError):
        compose_kernel(d[:, :, :, :0], p)
    d[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        compose_kernel(d, p)


def test_lift_preserves_initial_outputs_without_mutating_parent():
    import keras
    from dtr.models import research_student
    keras.utils.set_random_seed(47)
    parent = research_student(3, 64, 'separable_context')
    weights = [a.copy() for a in parent.get_weights()]
    model = build(parent)
    x = np.random.default_rng(47).uniform(0, 255, (5, 64, 64, 3)).astype(np.float32)
    np.testing.assert_allclose(model(x, training=False), parent(x, training=False), rtol=0, atol=1e-5)
    for a, b in zip(weights, parent.get_weights(), strict=True):
        np.testing.assert_array_equal(a, b)
    assert model.count_params() == cost()['parameters'] == 15667
    assert cost()['added_parameters'] == model.count_params() - parent.count_params()
    assert cost()['parent_macs'] == 895360
    assert cost()['macs'] == 1401216
    with pytest.raises(ValueError):
        build(research_student(3, 64, 'spatial'))
