import cv2
import numpy as np

from dtr.tracking import iou
from scripts.research_balloon_search import experimental, opening_split_candidates


def test_opening_recovers_balloon_from_thin_bridge_to_clutter():
    rgb = np.full((240, 320, 3), 20, np.uint8)
    cv2.circle(rgb, (90, 100), 30, (10, 20, 220), -1)
    cv2.rectangle(rgb, (118, 97), (180, 103), (10, 20, 220), -1)
    cv2.rectangle(rgb, (175, 80), (215, 120), (10, 20, 220), -1)
    found = opening_split_candidates(rgb)
    assert any(iou(c["box"], [60,70,121,131]) >= .8 for c in found)
    baseline = experimental(rgb, "mser")
    result = experimental(rgb, "mser_opening_split")
    assert result[:len(baseline)] == baseline
    assert len(result) <= min(12, len(baseline)+4)


def test_opening_has_no_candidates_on_neutral_scene():
    assert opening_split_candidates(np.full((240, 320, 3), 200, np.uint8)) == []


def test_opening_bounds_at_image_edges():
    rgb = np.full((240, 320, 3), 20, np.uint8)
    cv2.circle(rgb, (0, 100), 30, (220, 20, 10), -1)
    rows = opening_split_candidates(rgb)
    assert rows
    for row in rows:
        for key in ("box", "crop_box"):
            x, y, z, w = row[key]
            assert 0 <= x < z <= 320 and 0 <= y < w <= 240
