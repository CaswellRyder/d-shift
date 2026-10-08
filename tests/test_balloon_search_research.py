import numpy as np
import pytest

from scripts.research_balloon_search import experimental, remove_contained, select, suppress_confirmed_balloon_parts


def test_confirmed_parts_require_accepted_same_color_parent_and_do_not_mutate():
    parent = dict(box=[0, 0, 100, 100], label="red_balloon", accepted=True, score=.9)
    child = dict(box=[10, 10, 20, 20], label="red_balloon", accepted=True, score=.99)
    assert not suppress_confirmed_balloon_parts([parent, child])[1]["accepted"]
    assert child["accepted"]  # Keep original evidence intact.
    for bad_parent in (dict(parent, accepted=False), dict(parent, label="blue_balloon"),
                       dict(parent, label="background")):
        assert suppress_confirmed_balloon_parts([bad_parent, child])[1]["accepted"]
    neighbor = dict(child, box=[90, 90, 110, 110])
    assert suppress_confirmed_balloon_parts([parent, neighbor])[1]["accepted"]
    big_overlap = dict(child, box=[10, 10, 90, 90])
    assert suppress_confirmed_balloon_parts([parent, big_overlap])[1]["accepted"]
    goals = [dict(row, label="orange_circle_goal") for row in (parent, child)]
    assert all(row["accepted"] for row in suppress_confirmed_balloon_parts(goals))


@pytest.mark.parametrize("variant", ["baseline", "opened", "multisat", "mser", "mser_blend", "mser_components", "mser_confirmed_parts"])
def test_search_is_bounded_and_blank_scene_stays_empty(variant):
    frame = np.zeros((240, 320, 3), np.uint8)
    assert experimental(frame, variant) == []
    frame[40:70, 40:70] = (230, 0, 0)
    frame[40:70, 140:170] = (0, 0, 230)
    found = experimental(frame, variant)
    assert 1 <= len(found) <= 12
    assert all(c["color_group"] in (0, 1) for c in found)
    assert {c["color_group"] for c in found} == {0, 1}


def test_selector_deduplicates_and_does_not_starve_either_color():
    candidates = [dict(box=[i*10, 0, i*10+5, 5], color_group=i % 2,
                       proposal_score=100-i) for i in range(20)]
    found = select(candidates+candidates, limit=12)
    assert len(found) == 12
    assert len({tuple(c["box"]) for c in found}) == 12
    assert sum(c["color_group"] == 0 for c in found) == 6


def test_research_resolution_is_explicit():
    with pytest.raises(ValueError, match="320x240"):
        experimental(np.zeros((480, 640, 3), np.uint8), "mser")


def test_components_remove_same_color_interior_but_keep_other_color_and_neighbors():
    large = dict(box=[10, 10, 100, 100], color_group=0)
    inner = dict(box=[20, 20, 25, 25], color_group=0)
    other = dict(inner, color_group=1)
    neighbor = dict(box=[101, 10, 130, 40], color_group=0)
    kept = remove_contained([inner, neighbor, other, large])
    assert large in kept and other in kept and neighbor in kept
    assert inner not in kept
