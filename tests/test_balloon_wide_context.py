import numpy as np
import pytest

from scripts.balloon_wide_context import build, cost


def test_identity_initialization_and_independent_weights(tmp_path):
    keras = pytest.importorskip('keras')
    from dtr.models import research_student
    parent = research_student(3, variant='separable_context')
    before = [w.copy() for w in parent.get_weights()]
    model = build(parent)
    x = np.random.default_rng(47).uniform(0, 255, (3, 64, 64, 3)).astype(np.float32)
    np.testing.assert_array_equal(model(x).numpy(), parent(x).numpy())
    assert model.count_params() == 8563
    path = tmp_path/'student.keras'
    model.save(path)
    restored = keras.models.load_model(path, compile=False)
    np.testing.assert_array_equal(restored(x).numpy(), parent(x).numpy())
    delta = model.get_layer('wide_context_delta')
    delta.set_weights([np.ones_like(delta.get_weights()[0])*.01])
    assert not np.array_equal(model(x).numpy(), parent(x).numpy())
    for old, current in zip(before, parent.get_weights(), strict=True):
        np.testing.assert_array_equal(old, current)


def test_cost_and_reject_wrong_architecture():
    pytest.importorskip('keras')
    from dtr.models import research_student
    with pytest.raises(ValueError):
        build(research_student(3, variant='tiny'))
    report = cost()
    assert report['added_macs'] == 51200
    assert report['macs'] == report['parent_macs'] + 51200
    assert report['prepool_receptive_field'] == 63
