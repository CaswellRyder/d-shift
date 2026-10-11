import numpy as np
import pytest

from scripts.audit_balloon_dense_proposals import supplement
from scripts.research_blue_box_nms import decode
from scripts.research_blue_peak_blend import decode as parent


def grid():
    v = np.zeros((15, 20, 6), np.float32)
    v[..., :2] = -10
    v[..., 2:] = [.5, .5, .025, .025]
    return v


def test_nonpeak_distinct_box_survives_without_extra_inference_or_input_mutation():
    v = grid()
    v[5, 5, 1], v[5, 6, 1] = 1, .5
    saved = v.copy()
    assert len(parent(v, True)) == 1
    assert len(decode(v)) == 2
    assert decode(v)[0] == parent(v, True)[0]
    np.testing.assert_array_equal(v, saved)


def test_overlapping_nonpeak_is_suppressed_and_red_control_is_identical():
    v = grid()
    v[..., 4:] = [.4, .6]
    v[5, 5, 1], v[5, 6, 1] = 1, .5
    assert decode(v) == parent(v, True)
    v = grid()
    v[5, 5, 0], v[5, 6, 0] = 1, .5
    assert decode(v) == parent(v, True)


def test_finite_shape_geometry_and_clipped_boundary_contract():
    for invalid in (np.zeros((20, 15, 6)), np.full((15, 20, 6), np.nan)):
        with pytest.raises(ValueError):
            decode(invalid)
    v = grid()
    v[0, 0, 1], v[0, 1, 1] = 1, .5
    for row in decode(v):
        a, b, c, d = row['crop_box']
        assert 0 <= a < c <= 320 and 0 <= b < d <= 240
    v[0, 1, 2] = -1
    with pytest.raises(ValueError):
        decode(v)


def test_stable_bounded_decoder_and_unchanged_two_crop_supplement():
    v = grid()
    v[..., 1] = np.arange(300).reshape(15, 20)/100
    rows = decode(v)
    assert len(rows) == 12
    assert rows == decode(v)
    assert len(supplement([], rows)) == 2
    base = [dict(box=[0, 0, 1, 1]) for _ in range(11)]
    assert len(supplement(base, rows)) == 12
