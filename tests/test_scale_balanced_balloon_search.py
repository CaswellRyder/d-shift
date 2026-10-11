"""Bounded, deterministic scale allocation without synthetic accuracy claims."""
from copy import deepcopy

from scripts.scale_balanced_balloon_search import balanced_select


def rows(group, size, offset=0, count=4):
    return [dict(box=[i*50, offset, i*50+size, offset+size], color_group=group,
                 proposal_score=100-i) for i in range(count)]


def test_empty():
    assert balanced_select([]) == []


def test_four_buckets_get_three_slots_and_do_not_mutate_input():
    pool = rows(0, 16) + rows(1, 16, 50) + rows(0, 40, 100) + rows(1, 40, 150)
    saved = deepcopy(pool)
    result = balanced_select(pool)
    assert len(result) == 12
    for group in (0, 1):
        for small in (False, True):
            assert sum(r['color_group'] == group and (r['box'][2]-r['box'][0] <= 32) == small for r in result) == 3
    assert pool == saved
    assert balanced_select(pool) == result


def test_unused_quota_falls_back_without_underfilling():
    pool = rows(0, 16, count=20)
    assert len(balanced_select(pool)) == 12


def test_duplicates_never_consume_extra_slots():
    pool = rows(0, 16)
    assert len(balanced_select(pool*20)) == 4


def test_small_boundary_is_inclusive():
    pool = rows(0, 32) + rows(1, 33, 50)
    result = balanced_select(pool)
    assert result[0]['color_group'] == 0
    assert result[1]['color_group'] == 1
    assert len(result) == 8
