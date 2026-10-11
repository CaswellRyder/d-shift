from copy import deepcopy

import pytest

from scripts.research_balloon_confidence_selection import confidence_selection, concentric


def row(box, score=.9, label='blue_balloon', accepted=True):
    return dict(box=box, crop_box=box, score=score, label=label, accepted=accepted)


def test_confident_child_wins_without_mutating_scores_or_inputs():
    found = [row([24, 68, 196, 147], .98), row([51, 77, 158, 136], .99)]
    before = deepcopy(found)
    result = confidence_selection(found)
    assert found == before
    assert [r['accepted'] for r in result] == [False, True]
    assert [r['raw_accepted'] for r in result] == [True, True]
    assert result[0]['suppressed_by'] == 1


def test_confident_parent_and_ties_have_stable_winners():
    a, b = row([0, 0, 100, 100]), row([15, 15, 85, 85])
    assert concentric(a['box'], b['box'])
    assert [r['accepted'] for r in confidence_selection([a, b])] == [True, False]
    assert [r['accepted'] for r in confidence_selection([b, a])] == [True, False]


def test_different_colors_and_separate_instances_survive():
    found = [row([0, 0, 30, 30]), row([1, 1, 31, 31], label='red_balloon'), row([40, 0, 70, 30])]
    assert all(r['accepted'] for r in confidence_selection(found))


def test_rejected_parent_cannot_suppress_and_no_resurrection():
    result = confidence_selection([row([0, 0, 100, 100], .99, accepted=False), row([30, 30, 50, 50])])
    assert [r['accepted'] for r in result] == [False, True]


def test_reject_invalid_or_already_selected_inputs():
    for invalid in (row([0, 0, 0, 10]), row([0, 0, 10, 10], float('nan')),
                    row([0, 0, 10, 10], label='yellow_circle'),
                    dict(row([0, 0, 10, 10]), raw_accepted=True)):
        with pytest.raises(ValueError):
            confidence_selection([invalid])
