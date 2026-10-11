import numpy as np
import pytest

from scripts.balloon_center_geometry_adapter import network, joint_loss, frozen_weights


def test_joint_branch_initial_parity_and_frozen_parent():
    pytest.importorskip('tensorflow')
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
    target[:, 7, 10, 1:4] = [1, 1, 1]
    target[:, 7, 10, 4:9] = [.5, .5, .1, .1, 1]
    model.compile(optimizer=keras.optimizers.Adam(.001), loss=joint_loss)
    model.train_on_batch(image, target)
    after = model(image, training=False).numpy()
    assert frozen_weights(model) == frozen
    assert not np.array_equal(after[..., :2], before[..., :2])
    assert not np.array_equal(after[..., 2:], before[..., 2:])
    assert np.isfinite(after).all()
    assert model.count_params() == 3060
