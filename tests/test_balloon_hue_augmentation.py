import cv2
import numpy as np
import pytest

from scripts.balloon_hue_augmentation import recolor


@pytest.mark.parametrize('label,hues', [(1, [0, 10, 170]), (2, [90, 110, 130])])
def test_maps_foreground_to_correct_hue_and_preserves_white(label, hues):
    hsv = np.array([[*[ [h, 180, 180] for h in hues], [0, 0, 255], [30, 200, 200]]], np.uint8)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32)
    before = rgb.copy()
    out, target, mask = recolor(rgb, label)
    assert target == 3-label
    assert mask.tolist() == [[True, True, True, False, False]]
    np.testing.assert_array_equal(rgb, before)
    np.testing.assert_array_equal(out[~mask], rgb[~mask])
    changed = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_RGB2HSV)[..., 0][mask].astype(float)
    delta = (changed+90)%180-90 if target == 1 else changed-110
    assert np.max(np.abs(delta)) <= 6


@pytest.mark.parametrize('label', [0, -1, 3])
def test_rejects_background_and_unknown_labels(label):
    with pytest.raises(ValueError):
        recolor(np.zeros((2, 2, 3), np.uint8), label)


def test_rejects_invalid_pixels():
    with pytest.raises(ValueError):
        recolor(np.full((2, 2, 3), np.nan, np.float32), 1)
