import importlib.util
from pathlib import Path
import pytest


@pytest.fixture
def module(monkeypatch):
    scripts = Path(__file__).parents[1]/'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('verify_supplement', scripts/'verify_orange_supplement.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_same_class_one_to_one_scoring(module):
    target = dict(label='Orange Square Goal', box=[0, 0, 10, 10])
    prediction = dict(label='orange_square', box=[0, 0, 10, 10], accepted=True)
    assert module.counts([prediction, prediction], [target], 'orange_square') == dict(tp=1, fp=1, fn=0)
    assert module.counts([{**prediction, 'label': 'orange_circle'}], [target], 'orange_square') == dict(
        tp=0, fp=0, fn=1)
    assert module.counts([{**prediction, 'accepted': False}], [target], 'orange_square') == dict(
        tp=0, fp=0, fn=1)


def test_supplements_preserve_baseline_and_reject_conflicts(module):
    base = dict(box=[0, 0, 10, 10], accepted=True, label='orange_circle', score=.81)
    bigger = dict(box=[0, 0, 20, 20], accepted=True, label='orange_square', score=.99)
    distant = dict(box=[30, 30, 40, 40], accepted=True, label='orange_square', score=.96)
    assert module.admit_supplements([base], [bigger, distant], .95) == [base, distant]
    assert base['accepted']


def test_supplement_threshold_color_and_duplicate_guards(module):
    row = dict(box=[0, 0, 10, 10], accepted=True, label='orange_square', score=.91)
    assert module.admit_supplements([], [row], .95) == []
    assert module.admit_supplements([], [row, row], .9) == [row]
    assert module.admit_supplements([], [{**row, 'label': 'yellow_square'}], .9) == []
    assert module.admit_supplements([], [{**row, 'accepted': False}], .9) == []
    with pytest.raises(ValueError):
        module.admit_supplements([], [], 1.1)
