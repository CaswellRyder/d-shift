from collections import Counter

import numpy as np
import pytest

from scripts.indoor5_trial_data import choose
from scripts import train_indoor5_trial as trial
from scripts import export_indoor5_trial as exporter


def examples():
    return [dict(label=c, preserve=True, source_image=f'{c}/{i}')
            for c, n in [('background', 11), ('red_balloon', 32), ('blue_balloon', 25)] for i in range(n)]


def test_same_class_order_dose_and_protection_for_both_arms():
    extra = examples()
    original = [dict(r, source_image='old/'+r['source_image']) for r in extra]
    control, admit = choose(original, extra, 'control'), choose(original, extra, 'admit')
    assert len(control) == len(admit) == 136
    assert [r['label'] for r in control] == [r['label'] for r in admit]
    assert control == choose(original, extra, 'control')
    assert all(not r['preserve'] and r['view'] == 'original' for r in control + admit)
    assert Counter(r['source_image'] for r in admit) == Counter({r['source_image']: 2 for r in extra})
    with pytest.raises(ValueError):
        choose([], extra, 'control')
    with pytest.raises(ValueError):
        choose(original, extra[:-1], 'admit')


def test_indoor_gates_cannot_erase_previous_failures(monkeypatch):
    labels = np.r_[np.zeros(11, int), np.ones(32, int), np.full(25, 2)]
    logits = np.eye(3)[labels] * 5
    arrays = dict(indoor_labels=labels, indoor_after=logits)
    arrays.update({k: np.zeros((1922, 3)) for k in ('before', 'after', 'labels', 'ordinary')})
    monkeypatch.setattr(trial, 'prior_calculate', lambda _: {'training_gate_pass': False})
    assert trial.calculate(arrays)['indoor_passed'] == 68
    assert not trial.calculate(arrays)['training_gate_pass']
    monkeypatch.setattr(trial, 'prior_calculate', lambda _: {'training_gate_pass': True})
    assert trial.calculate(arrays)['training_gate_pass']
    for i in (0, 11, 43):
        changed = dict(arrays, indoor_after=logits.copy())
        changed['indoor_after'][i] = [0., 0., 0.]
        if i == 0:
            changed['indoor_after'][i, 2] = 5
        assert trial.calculate(changed)['indoor_passed'] == 67
        assert not trial.calculate(changed)['training_gate_pass']


def test_no_export_after_failed_training_gate(monkeypatch):
    monkeypatch.setattr(exporter, 'read_json', lambda _: {'eligible_for_development': False})
    with pytest.raises(ValueError, match='Require frozen paired TRAIN gate'):
        exporter.export()
