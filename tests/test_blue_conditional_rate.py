import pytest

from dtr.data import read_json
from scripts import train_blue_conditional_rate as rate
from scripts import train_blue_conditional_trial as original
from scripts import export_blue_conditional_rate as exporter
from scripts.summarize_blue_conditional_rate import variant_gates
from tests.test_blue_conditional_decision import comparisons


def test_only_learning_rate_changes_training_recipe():
    before, after = read_json(original.PROTOCOL), read_json(rate.PROTOCOL)
    assert after['learning_rate'] == 10 * before['learning_rate']
    assert after['learning_rate'] == .001
    for key in ('learning_rate', 'matched_recipe', 'rationale'):
        before.pop(key, None)
        after.pop(key, None)
    assert before == after
    assert rate.inputs is original.inputs
    assert rate.build is original.build
    assert rate.all_correct_loss is original.all_correct_loss


def test_failed_rate_trial_cannot_export(monkeypatch):
    monkeypatch.setattr(exporter, 'read_json', lambda _: {'eligible_for_development': False})
    with pytest.raises(ValueError, match='Require frozen paired TRAIN gate'):
        exporter.export()


def test_rate_scene_gate_requires_target_identity_retention():
    rows = comparisons()
    rows[0]['lost_parent_targets'] = ['fixture']
    assert not variant_gates(rows)['admit47/baseline']['nonregression_all_panels']
