import numpy as np
import pytest

from scripts.research_balloon_search import experimental, select


@pytest.mark.parametrize("variant", ["baseline", "opened", "multisat", "mser", "mser_blend"])
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
