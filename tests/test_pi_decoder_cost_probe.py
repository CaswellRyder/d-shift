import copy

import pytest

from scripts.pi_decoder_cost_probe import compare


def row():
    return dict(box=[0., 0., 8., 9.], crop_box=[0, 0, 9, 10], color_group=1, proposal_score=.6)


def test_parity_accepts_bounded_float_roundoff_only():
    original, changed = [row()], [row()]
    changed[0]['box'][2] += .00001
    assert compare(changed, original) < .001
    changed[0]['crop_box'][2] += 1
    with pytest.raises(ValueError):
        compare(changed, original)


def test_rescue_and_candidate_count_must_match():
    original = [row()]
    with pytest.raises(ValueError):
        compare([], original)
    changed = copy.deepcopy(original)
    changed[0]['neighbor_rescue'] = True
    with pytest.raises(ValueError):
        compare(changed, original)
