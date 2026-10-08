import cv2
import numpy as np
import pytest

from scripts.research_balloon_search import experimental
from scripts.evaluate_reviewed_balloon_scenes import evaluate


def test_chromatic_search_bounded_finite_and_input_unchanged():
    rgb = np.zeros((240, 320, 3), np.uint8)
    cv2.circle(rgb, (90, 100), 25, (6, 10, 60), -1)
    cv2.circle(rgb, (240, 100), 30, (80, 5, 8), -1)
    original = rgb.copy()
    found = experimental(rgb, "mser_chromatic")
    assert 0 < len(found) <= 12
    assert {c["color_group"] for c in found} == {0, 1}
    for candidate in found:
        a, b, c, d = candidate["crop_box"]
        assert 0 <= a < c <= 320 and 0 <= b < d <= 240
        assert np.isfinite(candidate["proposal_score"])
    np.testing.assert_array_equal(rgb, original)


def test_black_and_neutral_frames_do_not_gain_color():
    for value in (0, 29, 100, 255):
        assert experimental(np.full((240, 320, 3), value, np.uint8), "mser_chromatic") == []


def test_unknown_evaluation_search_rejected_before_io():
    with pytest.raises(ValueError, match="search variants"):
        evaluate("absent", "absent", {}, ("unknown",))


def test_blue_red_search_finds_cyan_without_creating_neutral_regions():
    rgb = np.full((240, 320, 3), 40, np.uint8)
    cv2.circle(rgb, (100, 100), 20, (20, 150, 155), -1)
    original = rgb.copy()
    assert experimental(rgb, "mser") == []
    found = experimental(rgb, "mser_blue_red")
    assert 0 < len(found) <= 12
    assert all(c["color_group"] == 1 for c in found)
    np.testing.assert_array_equal(rgb, original)
    for value in (0, 29, 100, 255):
        assert experimental(np.full_like(rgb, value), "mser_blue_red") == []
