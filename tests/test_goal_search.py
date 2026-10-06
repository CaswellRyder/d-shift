import cv2
import numpy as np
import pytest

from dtr.goal_search import edge_candidates, mixed_candidates
from dtr.tracking import iou
from dtr.vision import proposals


def gray_goal():
    rgb = np.full((240, 320, 3), 30, np.uint8)
    cv2.rectangle(rgb, (70, 70), (115, 115), (200, 200, 200), 3)
    return rgb


def test_neutral_goal_reaches_classifier_without_passing_hsv():
    rgb = gray_goal()
    assert proposals(rgb, "goal", profile="balloon_components") == []
    got = proposals(rgb, "goal", profile="goal_edges")
    assert 0 < len(got) <= 4
    assert max(iou(r["box"], [68, 68, 118, 118]) for r in got) >= .5
    assert all(r["color_group"] == -1 and "label" not in r for r in got)


@pytest.mark.parametrize("limit", [1, 2, 3, 6, 12, 24])
def test_existing_budget_is_not_increased(limit):
    rgb = gray_goal()
    baseline = [dict(box=[i*4, 2, i*4+3, 5], color_group=0) for i in range(limit)]
    result = mixed_candidates(rgb, baseline, limit)
    assert len(result) <= limit
    assert result[:limit-min(4, limit//3)] == baseline[:limit-min(4, limit//3)]


def test_blank_frame_has_no_edges_and_balloon_profile_unchanged():
    rgb = np.zeros((240, 320, 3), np.uint8)
    assert edge_candidates(rgb) == []
    cv2.circle(rgb, (100, 100), 20, (30, 220, 30), -1)
    assert proposals(rgb, "balloon", profile="goal_edges") == proposals(
        rgb, "balloon", profile="balloon_components")


def test_reject_unbounded_search_and_budget():
    with pytest.raises(ValueError, match="320x240"):
        edge_candidates(np.zeros((1944, 2592, 3), np.uint8))
    with pytest.raises(ValueError, match="budget"):
        edge_candidates(gray_goal(), 5)
