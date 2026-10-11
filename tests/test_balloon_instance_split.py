import cv2
import numpy as np

from dtr.tracking import iou
from scripts.research_balloon_search import experimental, instance_split_candidates


def test_touching_same_color_balloons_split_without_removing_baseline():
    rgb = np.full((240, 320, 3), 30, np.uint8)
    for center in ((100, 100), (155, 100)):
        cv2.circle(rgb, center, 30, (15, 25, 200), -1)
    original = rgb.copy()
    found = instance_split_candidates(rgb)
    for box in ([70, 70, 131, 131], [125, 70, 186, 131]):
        assert max((iou(box, c["box"]) for c in found), default=0) >= .5
    base = experimental(rgb, "mser")
    result = experimental(rgb, "mser_instance_split")
    assert result[:len(base)] == base and len(result) <= min(12, len(base)+4)
    np.testing.assert_array_equal(rgb, original)


def test_single_balloon_and_highlight_do_not_trigger_splitting():
    rgb = np.full((240, 320, 3), 30, np.uint8)
    cv2.circle(rgb, (100, 100), 40, (15, 25, 200), -1)
    cv2.circle(rgb, (90, 85), 8, (255, 255, 255), -1)
    assert instance_split_candidates(rgb) == []
    assert instance_split_candidates(np.full_like(rgb, 255)) == []


def test_split_regions_never_leave_image_bounds_at_edges():
    rgb = np.full((240, 320, 3), 30, np.uint8)
    for center in ((5, 100), (55, 100)):
        cv2.circle(rgb, center, 30, (200, 15, 25), -1)
    for row in instance_split_candidates(rgb):
        for key in ("box", "crop_box"):
            x, y, z, w = row[key]
            assert 0 <= x < z <= 320 and 0 <= y < w <= 240
