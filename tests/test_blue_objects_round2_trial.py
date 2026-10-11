from collections import Counter

import numpy as np
import pytest

from scripts.blue_objects_round2_trial_data import choose
from scripts import train_blue_objects_round2_trial as trial
from scripts import export_blue_objects_round2_trial as exporter


def test_equal_dose_background_controls():
    extra = [dict(label='background', preserve=False, source_image=f'new/{i}') for i in range(12)]
    old = [dict(r, preserve=True, source_image=f'old/{i}') for i, r in enumerate(extra)]
    control, admit = choose(old, extra, 'control'), choose(old, extra, 'admit')
    assert len(control) == len(admit) == 24
    assert control == choose(old, extra, 'control')
    assert all(r['label'] == 'background' and not r['preserve'] for r in control + admit)
    assert Counter(r['source_image'] for r in admit) == Counter({r['source_image']: 2 for r in extra})
    with pytest.raises(ValueError):
        choose([], extra, 'control')
    with pytest.raises(ValueError):
        choose(old, extra[:-1], 'admit')


def test_new_background_gate_cannot_override_prior_failures(monkeypatch):
    arrays = dict(object_after=np.tile([5., 0., 0.], (12, 1)))
    arrays.update({k: np.zeros((1946, 3)) for k in ('before', 'after', 'labels', 'ordinary')})
    monkeypatch.setattr(trial, 'prior_calculate', lambda _: {'training_gate_pass': False})
    assert trial.calculate(arrays)['object_background_correct'] == 12
    assert not trial.calculate(arrays)['training_gate_pass']
    monkeypatch.setattr(trial, 'prior_calculate', lambda _: {'training_gate_pass': True})
    assert trial.calculate(arrays)['training_gate_pass']
    arrays['object_after'][0] = [0., 0., 5.]
    assert not trial.calculate(arrays)['training_gate_pass']


def test_no_export_after_failed_training_gate(monkeypatch):
    monkeypatch.setattr(exporter, 'read_json', lambda _: {'eligible_for_development': False})
    with pytest.raises(ValueError, match='Require frozen paired TRAIN gate'):
        exporter.export()
