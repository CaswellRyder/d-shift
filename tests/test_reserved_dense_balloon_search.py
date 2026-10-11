from copy import deepcopy

import pytest

from scripts.reserved_dense_balloon_search import reserve


def row(i, group=0):
    return dict(box=[i*10, 0, i*10+8, 8], color_group=group, proposal_score=.9)


def test_preserves_ten_and_adds_red_and_blue_without_mutation():
    base = [row(i) for i in range(12)]
    learned = [row(15), row(16, 1)]
    saved = deepcopy((base, learned))
    out = reserve(base, learned)
    assert len(out) == 12 and out[:10] == base[:10] and out[10:] == learned
    assert (base, learned) == saved


def test_no_new_boxes_keeps_all_baseline():
    base = [row(i) for i in range(12)]
    assert reserve(base, []) == base
    assert reserve(base, [row(0)]) == base


def test_one_new_box_refills_unused_slot():
    base = [row(i) for i in range(12)]
    assert reserve(base, [row(15)]) == base[:10]+[row(15), row(10)]


def test_sparse_scene_does_not_exceed_two_added_boxes():
    assert len(reserve([], [row(i) for i in range(12)])) == 2


def test_rejects_unbounded_inputs():
    with pytest.raises(ValueError):
        reserve([row(i) for i in range(13)], [])
