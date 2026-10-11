import numpy as np
import pytest

from scripts.probe_balloon_spatial_dependence import VIEWS, rgb_histogram, transform


def test_all_views_preserve_joint_rgb_histogram_and_input():
    image = np.random.default_rng(12).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    original = image.copy()
    for view in VIEWS:
        value = transform(image, view)
        assert value.shape == image.shape and value.dtype == image.dtype
        assert np.array_equal(rgb_histogram(image), rgb_histogram(value))
        assert np.array_equal(transform(image, view), value)
        assert np.array_equal(image, original)
        if view != 'original':
            assert not np.array_equal(value, image)


def test_tile_permutation_preserves_each_patch():
    image = np.zeros((64, 64, 3), np.uint8)
    for i in range(64):
        image[i//8*8:i//8*8+8, i%8*8:i%8*8+8] = [i, i+1, i+2]
    result = transform(image, 'tiles8_seed47')
    seen = set()
    for i in range(64):
        tile = result[i//8*8:i//8*8+8, i%8*8:i%8*8+8]
        assert np.all(tile == tile[0,0])
        seen.add(tuple(tile[0,0]))
    assert seen == {(i, i+1, i+2) for i in range(64)}


def test_reject_unknown_views_and_wrong_resolution():
    with pytest.raises(ValueError):
        transform(np.zeros((64,64,3),np.uint8), 'pixels_seed49')
    with pytest.raises(ValueError):
        transform(np.zeros((32,32,3),np.uint8), 'original')
