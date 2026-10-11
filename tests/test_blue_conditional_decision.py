import pytest

from scripts import export_blue_conditional_trial as exporter
from scripts.summarize_blue_conditional_trial import PANELS, SEEDS, SEARCHES, variant_gates


def comparisons():
    return [dict(panel=p, seed=s, search=m, nonregression_vs_parent=True,
                 nonregression_vs_control=True, lost_parent_targets=[], meets_target=True)
            for p in PANELS for s in SEEDS for m in SEARCHES]


def test_failed_training_gate_prevents_export(monkeypatch):
    monkeypatch.setattr(exporter, 'read_json', lambda _: {'eligible_for_development': False})
    with pytest.raises(ValueError, match='Require frozen paired TRAIN gate'):
        exporter.export()


def test_all_fixed_comparisons_required_once():
    rows = comparisons()
    assert all(v['meets_target_all_panels'] for v in variant_gates(rows).values())
    for invalid in ([], rows[:-1], rows + [rows[0]], rows[:-1] + [rows[0]]):
        with pytest.raises(ValueError):
            variant_gates(invalid)


def test_target_identity_loss_cannot_hide_in_aggregate_gain():
    rows = comparisons()
    rows[0]['lost_parent_targets'] = [{'source': 'fixture', 'target': {'label': 'blue_balloon'}}]
    result = variant_gates(rows)['admit47/baseline']
    assert not result['nonregression_all_panels']
    assert result['meets_target_all_panels']


def test_target_and_nonregression_are_independent():
    rows = comparisons()
    rows[0]['meets_target'] = False
    rows[1]['nonregression_vs_control'] = False
    result = variant_gates(rows)
    assert result['admit47/baseline']['nonregression_all_panels']
    assert not result['admit47/baseline']['meets_target_all_panels']
    assert not result['admit47/coarse47']['nonregression_all_panels']
