import pytest

from scripts.summarize_blue_view_recovery import PANELS, SEEDS, SEARCHES, variant_gates


def comparisons():
    return [dict(panel=p, seed=s, search=m, nonregression_vs_parent=True,
                 nonregression_vs_control=True, lost_parent_targets=[], meets_target=True)
            for p in PANELS for s in SEEDS for m in SEARCHES]


def test_requires_every_comparison_once():
    rows = comparisons()
    assert all(r['meets_target_all_panels'] for r in variant_gates(rows).values())
    for invalid in ([], rows[:-1], rows + [rows[0]], rows[:-1] + [rows[0]]):
        with pytest.raises(ValueError):
            variant_gates(invalid)


def test_identity_loss_fails_even_when_aggregate_passes():
    rows = comparisons()
    rows[0]['lost_parent_targets'] = [dict(source='example', target={'label': 'blue_balloon'})]
    gate = variant_gates(rows)['admit47/baseline']
    assert not gate['nonregression_all_panels']
    assert gate['meets_target_all_panels']


def test_recall_target_and_control_nonregression_are_separate():
    rows = comparisons()
    rows[0]['meets_target'] = False
    rows[1]['nonregression_vs_control'] = False
    gates = variant_gates(rows)
    assert gates['admit47/baseline']['nonregression_all_panels']
    assert not gates['admit47/baseline']['meets_target_all_panels']
    assert not gates['admit47/coarse47']['nonregression_all_panels']
