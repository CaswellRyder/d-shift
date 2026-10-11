import numpy as np
import pytest

from scripts.blue_batch_trial_data import choose
from scripts import train_blue_batch_trial as train


def test_equal_hard_label_dose_and_deterministic_controls():
    pool = [dict(label='blue_balloon', preserve=True, id=i, view='original') for i in range(30)]
    extra = [dict(label='blue_balloon', preserve=True, id=100+i, view='original') for i in range(11)]
    control, admit = choose(pool, extra, 'control'), choose(pool, extra, 'admit')
    assert control == choose(pool, extra, 'control')
    assert len(control) == len(admit) == 22
    assert all(not r['preserve'] and r['view'] == 'original' for r in control + admit)
    assert control[:11] == control[11:] and admit[:11] == admit[11:]
    assert {r['id'] for r in control}.isdisjoint(r['id'] for r in admit)
    with pytest.raises(ValueError):
        choose(pool, extra, 'test')
    with pytest.raises(ValueError):
        choose(pool, extra[:-1], 'admit')


def test_every_new_positive_required_without_weakening_old_gate(monkeypatch):
    old_gate = True
    monkeypatch.setattr(train, 'prior_calculate', lambda a: {'training_gate_pass': old_gate})
    arrays = {k: np.zeros(1786) for k in ('before', 'after', 'labels', 'ordinary')}
    arrays['batch_after'] = np.tile([0., 0., 10.], (11, 1))
    assert train.calculate(arrays)['training_gate_pass']
    old_gate = False
    assert not train.calculate(arrays)['training_gate_pass']
    old_gate = True
    arrays['batch_after'][0] = [10., 0., 0.]
    result = train.calculate(arrays)
    assert result['batch_blue_accepted'] == 10 and not result['training_gate_pass']
