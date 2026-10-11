import copy

import pytest

from scripts.probe_window1_splits import supplement


def box(x, group=1, score=1.):
    return dict(box=[x, 0, x+10, 10], color_group=group, proposal_score=score)


def test_bounded_two_extras_without_mutating_baseline():
    base = [box(0)]
    old = copy.deepcopy(base)
    actual = supplement(base, [box(20, score=2), box(40, score=3), box(60)])
    assert actual == base+[box(40, score=3), box(20, score=2)]
    assert base == old


def test_no_duplicate_extra_but_other_color_allowed():
    base = [box(0)]
    assert supplement(base, [box(0), box(0, 0)]) == base+[box(0, 0)]
    assert supplement([], [box(0), box(0)]) == [box(0)]


def test_baseline_budget_fail_closed():
    with pytest.raises(ValueError):
        supplement([box(0)]*13, [])
