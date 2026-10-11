import numpy as np
import pytest

from scripts import train_blue_conditional_trial as trial
from scripts.blue_conditional_adapter import conditional_logits


def predictions():
    before = np.array([[0., 5., -2.], [4., -3., 2.]])
    after = conditional_logits(before, np.array([[5.], [-3.]]))
    return {prefix + suffix: value.copy() for prefix in trial.GROUPS
            for suffix, value in (('before', before), ('after', after))}


def test_keeps_old_gate_and_checks_every_probe_group(monkeypatch):
    monkeypatch.setattr(trial, 'prior_calculate', lambda a: {'training_gate_pass': False})
    arrays = predictions()
    report = trial.calculate(arrays)
    assert report['red_acceptance_unchanged'] and report['max_red_probability_delta'] < 1e-12
    assert not report['training_gate_pass']
    for prefix in trial.GROUPS:
        changed = {k: a.copy() for k, a in arrays.items()}
        changed[prefix + 'after'][0, 1] += .1
        with pytest.raises(ValueError, match='red probability'):
            trial.calculate(changed)


def test_weight_fingerprint_detects_values_and_dtypes():
    weights = [np.arange(4, dtype=np.float32).reshape(2, 2)]

    class Layer:
        def get_weights(self):
            return weights

    class Model:
        def get_layer(self, name):
            assert name == 'frozen'
            return Layer()

    before = trial.frozen_weights(Model(), ['frozen'])
    weights[0][0, 0] += 1
    assert trial.frozen_weights(Model(), ['frozen']) != before
    weights[0][0, 0] -= 1
    assert trial.frozen_weights(Model(), ['frozen']) == before
    weights[0] = weights[0].astype(np.float64)
    assert trial.frozen_weights(Model(), ['frozen']) != before
