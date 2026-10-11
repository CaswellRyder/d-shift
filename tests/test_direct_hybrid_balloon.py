from copy import deepcopy

import pytest

from scripts.direct_hybrid_balloon import merge
from scripts.research_confident_containment import select as select_crops


def row(box, score=.9, label='red_balloon', accepted=True):
    return dict(box=box, score=score, label=label, accepted=accepted)


@pytest.mark.parametrize('policy', ['pooled', 'base_first'])
def test_all_base_candidates_kept_and_novel_direct_added(policy):
    base = [row([i*15, 0, i*15+10, 10]) for i in range(12)]
    direct = [row([240, 180, 260, 200], .7)]
    original = deepcopy(base+direct)
    found = merge(base, direct, policy)
    assert len(found) == 13 and all(r['accepted'] for r in found)
    assert base+direct == original


def test_base_priority_and_pooled_scores_are_distinct():
    base, direct = [row([10, 10, 30, 30])], [row([11, 11, 31, 31], .99)]
    assert [r['accepted'] for r in merge(base, direct, 'base_first')] == [True, False]
    assert [r['accepted'] for r in merge(base, direct, 'pooled')] == [False, True]


def test_low_direct_probability_does_not_reclassify_low_crop_probability():
    found = merge([row([0, 0, 10, 10], .7, accepted=False)],
                  [row([40, 40, 60, 60], .6)], 'pooled')
    assert [r['accepted'] for r in found] == [False, True]
    assert [r['score'] for r in found] == [.7, .6]


def test_base_only_matches_original_selection():
    base = [row([0, 0, 20, 20]), row([1, 1, 21, 21], .95)]
    expected = select_crops([dict(r, proposal_origin='mser') for r in base])
    assert merge(base, [], 'base_first') == expected
    assert merge(base, [], 'pooled') == expected


def test_direct_suppression_indices_include_base_offset():
    base = [row([100, 100, 120, 120])]
    direct = [row([0, 0, 20, 20], .6), row([1, 1, 21, 21], .7)]
    found = merge(base, direct, 'base_first')
    assert found[1]['suppressed_by'] == 2
    assert [r['accepted'] for r in found] == [True, False, True]


def test_contracts_and_empty_input():
    with pytest.raises(ValueError):
        merge([], [], 'arbitrary')
    with pytest.raises(ValueError):
        merge([row([0, 0, 10, 10])] * 13, [], 'pooled')
    with pytest.raises(ValueError):
        merge([row([0, 0, 10, 10], .7)], [], 'pooled')
    with pytest.raises(ValueError):
        merge([], [row([0, 0, 10, 10], .4)], 'base_first')
    assert merge([], [], 'pooled') == []
