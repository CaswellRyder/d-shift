import numpy as np
import pytest

from dtr.runtime import softmax
from scripts.blue_conditional_adapter import build, conditional_logits


def test_zero_correction_and_red_mass_invariance_even_for_extreme_corrections():
    rng = np.random.default_rng(47)
    z = rng.normal(0, 10, (100, 3))
    original = softmax(z)
    np.testing.assert_allclose(softmax(conditional_logits(z, np.zeros((100, 1)))), original, atol=1e-14)
    for delta in (-1000., -10., 10., 1000.):
        result = softmax(conditional_logits(z, np.full((100, 1), delta)))
        np.testing.assert_allclose(result[:, 1], original[:, 1], rtol=1e-12, atol=1e-14)
        mask = original[:, 1] >= .8
        assert (result[mask, 1] >= .8).all()
        assert (result[mask].argmax(1) == 1).all()
        assert (result[:, 2] <= 1 - original[:, 1] + 1e-14).all()
    with pytest.raises(ValueError):
        conditional_logits(z, np.zeros(100))
    with pytest.raises(ValueError):
        conditional_logits(z, np.full((100, 1), np.nan))


def test_only_adapter_trainable_serialization_and_parent_unchanged(tmp_path):
    import keras
    from dtr.models import research_student
    keras.utils.set_random_seed(47)
    parent = research_student(3, 64, 'separable_context')
    saved = [w.copy() for w in parent.get_weights()]
    model = build(parent)
    x = np.random.default_rng(47).uniform(0, 255, (5, 64, 64, 3)).astype(np.float32)
    original = softmax(parent(x, training=False).numpy())
    np.testing.assert_allclose(softmax(model(x, training=False).numpy()), original, atol=1e-6)
    assert model.count_params() == 9844
    assert sum(int(np.prod(w.shape)) for w in model.trainable_weights) == 2081
    kernel, bias = model.get_layer('blue_adapter_delta').get_weights()
    model.get_layer('blue_adapter_delta').set_weights([kernel, bias + 9.])
    changed = model(x, training=False).numpy()
    np.testing.assert_allclose(softmax(changed)[:, 1], original[:, 1], atol=1e-6)
    assert all(np.array_equal(a, b) for a, b in zip(saved, parent.get_weights(), strict=True))
    path = tmp_path / 'adapter.keras'
    model.save(path)
    loaded = keras.models.load_model(path, compile=False)
    np.testing.assert_allclose(loaded(x, training=False), changed, atol=1e-6)
