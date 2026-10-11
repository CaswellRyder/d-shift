import numpy as np
import pytest

from scripts.balloon_geometry_adapter import network, frozen_weights, geometry_loss
from scripts.audit_geometry_adapter import gate
from copy import deepcopy


def test_zero_branch_reproduces_parent_and_training_cannot_change_heat():
    tf = pytest.importorskip('tensorflow')
    import keras
    from scripts.train_balloon_dense_proposals import network as original
    keras.utils.set_random_seed(47)
    parent = original()
    image = np.random.default_rng(47).uniform(0, 255, (2, 120, 160, 3)).astype(np.float32)
    before = parent(image, training=False).numpy()
    model = network(parent)
    assert np.allclose(model(image, training=False).numpy(), before, atol=1e-6, rtol=0)
    frozen = frozen_weights(model)
    target = np.zeros((2, 15, 20, 9), np.float32)
    target[:, 7, 10, 4:9] = [.5, .5, .1, .1, 1]
    model.compile(optimizer=keras.optimizers.Adam(.001), loss=geometry_loss)
    model.train_on_batch(image, target)
    after = model(image, training=False).numpy()
    assert frozen_weights(model) == frozen
    assert np.allclose(after[..., :2], before[..., :2], atol=1e-6, rtol=0)
    assert not np.array_equal(after[..., 2:], before[..., 2:])
    assert np.isfinite(after).all()
    assert model.count_params() == 3026
    empty = tf.constant(np.zeros_like(target))
    assert float(geometry_loss(empty, tf.constant(after))) == 0


def test_gate_requires_actual_bounded_coverage_and_paired_improvement():
    parent = {split: {color: dict(targets=10, learned=7, supplement=8)
                     for color in ('red_balloon', 'blue_balloon')}
              for split in ('base_fit', 'base_holdout', 'extra_fit')}
    current = deepcopy(parent)
    control = deepcopy(parent)
    current['extra_fit']['blue_balloon']['learned'] = 9
    control['extra_fit']['blue_balloon']['learned'] = 8
    assert gate(parent, current, control)
    current['base_fit']['blue_balloon']['supplement'] = 7
    assert not gate(parent, current, control)
    current['base_fit']['blue_balloon']['supplement'] = 8
    control['extra_fit']['blue_balloon']['learned'] = 9
    assert not gate(parent, current, control)
