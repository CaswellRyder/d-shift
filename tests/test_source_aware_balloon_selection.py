from copy import deepcopy

import pytest

from scripts.source_aware_balloon_selection import select
from scripts.research_confident_containment import select as incumbent


def row(box, source, score=.9, label='blue_balloon', accepted=True):
    return dict(box=box, proposal_origin=source, score=score, label=label, accepted=accepted)


def test_stronger_learned_box_cannot_evict_base():
    original = [row([10, 10, 30, 30], 'mser'), row([11, 11, 31, 31], 'learned', .99)]
    snapshot = deepcopy(original)
    found = select(original)
    assert [r['accepted'] for r in found] == [True, False]
    assert found[1]['score'] == .99 and found[1]['raw_accepted']
    assert found[1]['suppressed_by_index'] == 0
    assert original == snapshot


def test_nonconflicting_learned_targets_survive():
    found = select([row([10, 10, 30, 30], 'mser'), row([80, 80, 100, 100], 'learned')])
    assert all(r['accepted'] for r in found)


def test_unaccepted_base_cannot_hide_learned_target():
    found = select([row([10, 10, 30, 30], 'mser', .6, accepted=False),
                    row([10, 10, 30, 30], 'learned')])
    assert [r['accepted'] for r in found] == [False, True]


def test_different_colors_do_not_suppress_each_other():
    found = select([row([10, 10, 30, 30], 'mser', label='red_balloon'),
                    row([10, 10, 30, 30], 'learned')])
    assert all(r['accepted'] for r in found)


@pytest.mark.parametrize('source', ['mser', 'learned'])
def test_single_source_preserves_incumbent_selection(source):
    items = [row([10, 10, 30, 30], source), row([11, 11, 31, 31], source, .95)]
    assert select(items) == incumbent(items)


def test_local_suppression_indices_remapped_to_original_order():
    items = [row([100, 100, 120, 120], 'learned'), row([10, 10, 30, 30], 'mser'),
             row([11, 11, 31, 31], 'mser', .99)]
    found = select(items)
    assert found[1]['suppressed_by'] == 2
    assert [r['accepted'] for r in found] == [True, False, True]


def test_nested_learned_parent_cannot_evict_base_child():
    found = select([row([20, 20, 30, 30], 'mser'), row([0, 0, 80, 80], 'learned', .99)])
    assert [r['accepted'] for r in found] == [True, False]


def test_invalid_contracts_rejected():
    with pytest.raises(ValueError):
        select([row([0, 0, 10, 10], 'unknown')])
    with pytest.raises(ValueError):
        select([row([0, 0, 10, 10], 'mser')] * 13)
    with pytest.raises(ValueError):
        select(incumbent([row([0, 0, 10, 10], 'mser')]))
    assert select([]) == []
