import numpy as np
import pytest

from scripts.independent_neighbor_decoder import decode
from scripts.research_blue_peak_blend import decode as baseline


def grid():
    v = np.zeros((15, 20, 6), np.float32)
    v[..., :2] = -10
    v[..., 2:] = [.5, .5, .025, .04]
    return v


def test_distinct_adjacent_objects_are_rescued_without_changing_peaks():
    v = grid()
    v[5, 5, 0], v[5, 6, 0] = 1., .5
    saved = v.copy()
    old, new = baseline(v, True), decode(v)
    assert len(old) == 1 and len(new) == 2
    assert new[0] == old[0]
    assert new[1]['neighbor_rescue'] is True
    np.testing.assert_array_equal(v, saved)


def test_same_object_neighbor_or_weak_cell_is_not_rescued():
    v = grid()
    v[..., 4:] = [.2, .4]
    v[5, 5, 0], v[5, 6, 0] = 1., .5
    assert decode(v) == baseline(v, True)
    v = grid()
    v[5, 5, 0], v[5, 6, 0] = 1., -.01
    assert decode(v) == baseline(v, True)


def test_blue_blending_parity_when_no_distinct_rescue_exists():
    v = grid()
    v[..., 4:] = [.2, .4]
    v[5, 5, 1], v[5, 6, 1] = 1., .99
    assert decode(v) == baseline(v, True)


def test_clipped_boxes_and_budget_are_valid():
    v = grid()
    v[::2, ::2, 0] = 1.
    v[::2, 1::2, 0] = .5
    found = decode(v)
    assert len(found) == 12
    for r in found:
        a, b, c, d = r['crop_box']
        assert 0 <= a < c <= 320 and 0 <= b < d <= 240


def test_invalid_values_fail_closed():
    with pytest.raises(ValueError):
        decode(np.zeros((2, 2, 6), np.float32))
    v = grid()
    v[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        decode(v)
