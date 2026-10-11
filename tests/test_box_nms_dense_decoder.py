import numpy as np
import pytest

from scripts.box_nms_dense_decoder import decode
from scripts.research_blue_peak_blend import decode as peak_decode


def values():
    v = np.zeros((15, 20, 6), np.float32)
    v[..., :2] = -10
    return v


def test_adjacent_separate_red_targets_survive_box_nms():
    v = values()
    v[2, 2] = [5, -10, .5, .5, 8/320, 8/240]
    v[2, 3] = [4, -10, .5, .5, 8/320, 8/240]
    assert len(peak_decode(v, True)) == 1
    assert len(decode(v)) == 2


def test_isolated_peak_has_exact_old_decoder_parity():
    v = values()
    v[2, 2] = [-10, 5, .5, .5, 20/320, 20/240]
    assert decode(v) == peak_decode(v, True)


def test_overlapping_boxes_are_still_deduplicated():
    v = values()
    v[2, 2] = [5, -10, .5, .5, .4, .4]
    v[2, 3] = [4, -10, .5, .5, .4, .4]
    assert len(decode(v)) == 1


def test_stays_bounded_at_twelve():
    v = values()
    v[..., 0] = 5
    v[..., 2:4] = .5
    v[..., 4:] = .02
    assert len(decode(v)) == 12


@pytest.mark.parametrize('v', [np.zeros((2, 3, 6)), np.full((15, 20, 6), np.nan)])
def test_rejects_invalid_output(v):
    with pytest.raises(ValueError):
        decode(v)
