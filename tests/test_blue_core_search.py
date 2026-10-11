import cv2
import numpy as np
import pytest

from dtr.tracking import iou
from scripts.research_balloon_search import blue_core_candidates, experimental


def scene():
    return np.full((240,320,3), 25, np.uint8)


def test_thick_blue_center_survives_connection_to_long_thin_clutter():
    rgb = scene()
    cv2.ellipse(rgb, (100,100), (40,25), 0, 0, 360, (10,20,220), -1)
    cv2.rectangle(rgb, (125,110), (230,120), (10,20,220), -1)
    found = blue_core_candidates(rgb)
    assert any(iou(r["box"], [60,75,141,126]) >= .5 for r in found)
    assert all(r["color_group"] == 1 and r["proposal_source"] == "blue_distance_core" for r in found)


def test_two_cores_and_bounded_supplement_preserve_baseline():
    rgb = scene()
    cv2.circle(rgb, (80,100), 30, (10,20,220), -1)
    cv2.circle(rgb, (150,100), 30, (10,20,220), -1)
    cv2.rectangle(rgb, (100,95), (140,105), (10,20,220), -1)
    found = blue_core_candidates(rgb)
    for box in ([50,70,111,131], [120,70,181,131]):
        assert any(iou(r["box"],box) >= .5 for r in found)
    baseline = experimental(rgb, "mser")
    result = experimental(rgb, "mser_blue_cores")
    assert result[:len(baseline)] == baseline
    assert len(result) <= min(12,len(baseline)+2)


def test_neutral_and_red_only_frames_do_not_generate_blue_cores():
    rgb = scene()
    assert blue_core_candidates(rgb) == []
    cv2.circle(rgb, (80,100), 30, (220,20,10), -1)
    assert blue_core_candidates(rgb) == []


def test_highlights_are_filled_and_border_boxes_stay_in_frame():
    rgb = scene()
    cv2.circle(rgb, (5,100), 30, (10,20,220), -1)
    cv2.circle(rgb, (12,95), 3, (255,255,255), -1)
    found = blue_core_candidates(rgb)
    assert found
    for row in found:
        for key in ("box", "crop_box"):
            x,y,z,w = row[key]
            assert 0 <= x < z <= 320 and 0 <= y < w <= 240


def test_wrong_frame_shape_is_rejected():
    with pytest.raises(ValueError, match="320x240"):
        experimental(np.zeros((480,640,3), np.uint8), "mser_blue_cores")
