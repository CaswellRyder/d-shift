import cv2
import numpy as np
import pytest

from dtr.tracking import iou
from scripts.research_balloon_search import (
    blue_core_candidates, blue_radial_candidates, blue_ellipse_candidates,
    ellipse_core_box, experimental,
)


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


@pytest.mark.parametrize("angle", [0,30,60,90,135])
def test_radial_bounds_follow_elongated_balloon_at_different_angles(angle):
    rgb = scene()
    cv2.ellipse(rgb, (120,110), (55,22), angle, 0, 360, (10,20,220), -1)
    mask = (rgb[:,:,2] > 200).astype(np.uint8)
    x,y,w,h = cv2.boundingRect(mask)
    truth = [x,y,x+w,y+h]
    radial = blue_radial_candidates(rgb)
    assert radial and max(iou(r["box"],truth) for r in radial) > .8
    assert all(r["proposal_source"] == "blue_radial_core" for r in radial)


def test_radial_bounds_ignore_thin_bridge_and_keep_two_objects():
    rgb = scene()
    cv2.circle(rgb, (80,100), 30, (10,20,220), -1)
    cv2.circle(rgb, (160,100), 30, (10,20,220), -1)
    cv2.rectangle(rgb, (100,96), (140,104), (10,20,220), -1)
    for target in ([50,70,111,131], [130,70,191,131]):
        assert max(iou(r["box"],target) for r in blue_radial_candidates(rgb)) > .8


def test_radial_supplement_keeps_baseline_and_clips_border_boxes():
    rgb = scene()
    cv2.ellipse(rgb, (10,5), (30,18), 35, 0, 360, (10,20,220), -1)
    base, result = experimental(rgb, "mser"), experimental(rgb, "mser_blue_radial")
    assert result[:len(base)] == base and len(result) <= min(12,len(base)+2)
    for row in result:
        for key in ("box", "crop_box"):
            x,y,z,w = row[key]
            assert 0 <= x < z <= 320 and 0 <= y < w <= 240


def test_invalid_core_geometry_rejected():
    with pytest.raises(ValueError, match="core geometry"):
        blue_core_candidates(scene(), "unknown")


@pytest.mark.parametrize("angle", [0,30,60,90,135])
def test_robust_ellipse_localizes_rotated_oval(angle):
    rgb = scene()
    cv2.ellipse(rgb, (120,110), (55,22), angle, 0, 360, (10,20,220), -1)
    x,y,w,h = cv2.boundingRect((rgb[:,:,2] > 200).astype(np.uint8))
    found = blue_ellipse_candidates(rgb)
    assert found and max(iou(r["box"],[x,y,x+w,y+h]) for r in found) > .8
    assert all(r["proposal_source"] == "blue_ellipse_core" for r in found)


def test_robust_ellipse_handles_thin_bridge():
    rgb = scene()
    cv2.circle(rgb, (80,100), 30, (10,20,220), -1)
    cv2.circle(rgb, (160,100), 30, (10,20,220), -1)
    cv2.rectangle(rgb, (100,96), (140,104), (10,20,220), -1)
    for target in ([50,70,111,131], [130,70,191,131]):
        assert max(iou(r["box"],target) for r in blue_ellipse_candidates(rgb)) > .8


def test_ellipse_clips_border_and_preserves_baseline_budget():
    rgb = scene()
    cv2.ellipse(rgb, (10,5), (30,18), 35, 0, 360, (10,20,220), -1)
    base, result = experimental(rgb, "mser"), experimental(rgb, "mser_blue_ellipse")
    assert result[:len(base)] == base and len(result) <= min(12,len(base)+2)
    for row in result:
        for key in ("box", "crop_box"):
            x,y,z,w = row[key]
            assert 0 <= x < z <= 320 and 0 <= y < w <= 240


def test_ellipse_invalid_fit_falls_back_without_crashing():
    mask = np.zeros((242,322),np.uint8)
    assert ellipse_core_box(mask, 100,100,10) == (87,87,114,114)


def test_ellipse_ignores_neutral_frame_and_rejects_wrong_shape():
    assert blue_ellipse_candidates(scene()) == []
    with pytest.raises(ValueError, match="320x240"):
        experimental(np.zeros((480,640,3),np.uint8), "mser_blue_ellipse")
