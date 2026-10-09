import numpy as np

from scripts.research_pixel_photometric import photometric
from scripts.research_pixel_proposals import (BoxPixelMasks, CvPixelMasks, PixelMasks, component_candidate, coverage,
                                              pixel_search)

RED, BLUE = (178, 28, 48), (26, 86, 179)


def test_masks_match_squared_rgb_distance():
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
    red, blue = PixelMasks((RED, BLUE), 90)(rgb)
    for mask, ref in ((red, RED), (blue, BLUE)):
        distance = ((rgb.astype(np.int64) - ref) ** 2).sum(axis=2)
        np.testing.assert_array_equal(mask, (distance <= 90 ** 2).astype(np.uint8))


def test_component_candidate_keeps_mser_geometry_filters():
    shape = (240, 320, 3)
    c = component_candidate(100, 50, 25, 20, 400, 1, shape)
    assert c["box"] == [100, 50, 125, 70]
    assert c["crop_box"] == [97, 47, 128, 73]
    assert c["color_group"] == 1 and c["area"] == 400.
    assert component_candidate(0, 0, 40, 10, 400, 0, shape) is None  # aspect > 3
    assert component_candidate(0, 0, 3, 10, 30, 0, shape) is None  # too thin
    assert component_candidate(0, 0, 300, 220, 60000, 0, shape) is None  # most of the frame


def test_pixel_search_proposes_each_color_blob_and_skips_small_ones():
    rgb = np.zeros((240, 320, 3), np.uint8)
    rgb[20:50, 30:55] = RED
    rgb[100:130, 200:230] = BLUE
    rgb[200:205, 10:15] = RED  # 25 px, below the 53 px minimum
    proposals = pixel_search(rgb, PixelMasks((RED, BLUE), 75), 53)
    assert sorted((p["color_group"], p["box"]) for p in proposals) == [
        (0, [30, 20, 55, 50]), (1, [200, 100, 230, 130])]
    truth = [dict(label="red_balloon", box=[30, 20, 55, 50]),
             dict(label="blue_balloon", box=[30, 20, 55, 50])]
    assert coverage(truth, [p["box"] for p in proposals], [p["color_group"] for p in proposals]) == 1


def test_fast_masks_match_numpy_masks():
    rng = np.random.default_rng(1)
    rgb = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
    for threshold in (60, 75, 90, 105, 147):
        expected = PixelMasks((RED, BLUE), threshold)(rgb)
        for cls in (CvPixelMasks, BoxPixelMasks):
            for fast, slow in zip(cls((RED, BLUE), threshold)(rgb), expected):
                np.testing.assert_array_equal(fast != 0, slow != 0)


def test_photometric_identity_and_gain_clip():
    rgb = np.array([[[0, 128, 255], [10, 200, 250]]], np.uint8)
    np.testing.assert_array_equal(photometric(rgb, (1., 1., 1.), 1.), rgb)
    np.testing.assert_array_equal(photometric(rgb, (.5, 1., 1.25), 1.),
                                  [[[0, 128, 255], [5, 200, 255]]])
