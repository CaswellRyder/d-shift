import numpy as np
import pytest

from scripts.shared_proposal_quality import candidate, feature_row, gate, network


def row(origin='mser'):
    return dict(box=[0., 0., 16., 16.], color_group=0, proposal_score=32.,
                label='blue_balloon', score=.9, proposal_origin=origin, accepted=True)


def test_uses_classifier_probability_and_label_not_mser_contrast_or_search_color():
    actual = candidate(row())
    assert actual['color_group'] == 1 and actual['proposal_score'] == .9
    vector = feature_row(np.zeros((15, 20, 24), np.float32), row())
    np.testing.assert_allclose(vector[28:], [.9, 0, 1, 0, 1])
    assert vector.shape == (33,)


def test_fixed_gate_handles_both_sources_without_reactivating_rejections():
    rows = [row(), row('learned'), dict(row(), accepted=False)]
    result = gate(rows, [.49, .5, None])
    assert [r['accepted'] for r in result] == [False, True, False]
    assert result[2] == rows[2] and rows[0]['accepted'] is True
    with pytest.raises(ValueError):
        gate(rows, [.3, .4, .5])
    with pytest.raises(ValueError):
        gate([row()], [np.nan])


def test_small_head_budget_and_input_contract():
    assert network().count_params() == 281
    with pytest.raises(ValueError):
        candidate(dict(row(), label='background'))
    with pytest.raises(ValueError):
        candidate(dict(row(), score=32.))
