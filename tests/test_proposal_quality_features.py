import numpy as np
import pytest

from scripts.proposal_quality_features import features, quality_target, gate, network


def proposal():
    return dict(box=[0., 0., 16., 16.], color_group=0, proposal_score=.75)


def test_feature_geometry_and_pooling():
    maps = np.zeros((15, 20, 24), np.float32)
    maps[0, 0] = np.arange(24)
    result = features(maps, proposal())
    np.testing.assert_array_equal(result[:24], np.arange(24))
    np.testing.assert_allclose(result[24:], [8/320, 8/240, 16/320, 16/240, .75, 1, 0])
    assert result.shape == (31,) and result.dtype == np.float32


def test_quality_requires_complete_review_and_correct_color():
    truth = [dict(label='red_balloon', box=[0., 0., 16., 16.])]
    assert quality_target(proposal(), truth, True) == 1.
    assert quality_target(dict(proposal(), color_group=1), truth, True) == 0.
    with pytest.raises(ValueError):
        quality_target(proposal(), truth, False)


def test_gate_only_removes_accepted_learned_rows():
    learned = dict(proposal(), accepted=True, proposal_origin='learned')
    base = dict(learned, proposal_origin='mser')
    old_rejection = dict(learned, accepted=False, suppressed=True, suppression_reason='existing')
    actual = gate([base, learned, old_rejection], [None, .49, .9])
    assert actual[0] == base and actual[1]['accepted'] is False
    assert actual[2]['accepted'] is False and actual[2]['suppression_reason'] == 'existing'
    assert learned['accepted'] is True
    assert gate([learned], [.5])[0]['accepted'] is True
    with pytest.raises(ValueError):
        gate([base], [.2])
    with pytest.raises(ValueError):
        gate([learned], [np.nan])


def test_head_is_bounded():
    model = network()
    assert model.count_params() == 265 and model.output_shape == (None, 1)
