import cv2
import numpy as np
import pytest

from scripts import research_balloon_search as search


def test_levels_are_bounded_blue_only_and_preserve_pixels():
    rgb = np.full((240, 320, 3), 30, np.uint8)
    cv2.circle(rgb, (80, 90), 25, (10, 20, 200), -1)
    cv2.circle(rgb, (240, 90), 20, (220, 20, 10), -1)
    original = rgb.copy()
    rows = search.blue_level_candidates(rgb)
    assert 0 < len(rows) <= 12
    assert all(c["color_group"] == 1 for c in rows)
    base = search.experimental(rgb, "mser")
    supplemented = search.experimental(rgb, "mser_blue_levels")
    assert supplemented[:len(base)] == base
    assert len(supplemented) <= min(12, len(base) + 4)
    for row in supplemented:
        x, y, z, w = row["crop_box"]
        assert 0 <= x < z <= 320 and 0 <= y < w <= 240
    np.testing.assert_array_equal(rgb, original)


@pytest.mark.parametrize("value", [0, 15, 100, 255])
def test_neutral_frames_do_not_gain_blue(value):
    rgb = np.full((240, 320, 3), value, np.uint8)
    assert search.experimental(rgb, "mser_blue_levels") == []


def test_full_baseline_budget_never_grows_or_reorders(monkeypatch):
    rgb = np.zeros((240, 320, 3), np.uint8)
    base = [dict(box=[i*10, 0, i*10+5, 5], color_group=0, proposal_score=1.)
            for i in range(12)]
    monkeypatch.setattr(search, "select", lambda pool, limit=12: list(base))
    assert search.experimental(rgb, "mser_blue_levels") == base


def test_bad_resolution_rejected_before_search():
    with pytest.raises(ValueError, match="320x240"):
        search.experimental(np.zeros((480, 640, 3), np.uint8), "mser_blue_levels")
