import numpy as np
import pytest


def test_continuation_restores_optimizer_slots_and_rejects_wrong_iteration(tmp_path):
    pytest.importorskip('tensorflow')
    import keras
    from scripts.train_balloon_grid_convergence import optimizer_evidence
    keras.utils.set_random_seed(47)
    model = keras.Sequential([keras.layers.Input((2,)), keras.layers.Dense(1)])
    model.compile(optimizer=keras.optimizers.Adam(.001), loss='mse')
    model.train_on_batch(np.ones((2, 2), np.float32), np.zeros((2, 1), np.float32))
    before = [v.numpy().copy() for v in model.optimizer.variables]
    model.save(tmp_path / 'parent.keras')
    restored = keras.models.load_model(tmp_path / 'parent.keras')
    evidence = optimizer_evidence(restored, 1)
    assert evidence['iterations'] == 1 and evidence['nonzero_slots']
    for old, new in zip(before, restored.optimizer.variables, strict=True):
        np.testing.assert_array_equal(old, new.numpy())
    with pytest.raises(ValueError, match='not restored'):
        optimizer_evidence(restored, 2)
    restored.train_on_batch(np.ones((2, 2), np.float32), np.zeros((2, 1), np.float32))
    assert optimizer_evidence(restored, 2)['iterations'] == 2


def test_convergence_data_arm_must_be_explicit(monkeypatch):
    from scripts import train_balloon_grid_convergence as module
    monkeypatch.setattr(module, 'read_json', lambda _: dict(data='/unused', manifest_sha256='expected'))
    monkeypatch.setattr(module, 'sha256', lambda _: 'expected')
    with pytest.raises(ValueError, match='Unknown grid arm'):
        module.source_data('unreviewed')
