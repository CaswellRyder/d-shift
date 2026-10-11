import numpy as np
import pytest

from scripts.direct_localizer_balloon import detect, select_direct
from scripts.research_confident_containment import select as incumbent_select


def values(logit, group=0):
    v = np.full((15, 20, 6), .5, np.float32)
    v[..., :2] = -10
    v[2, 2, 2:] = [0, 0, 16/320, 16/240]
    v[2, 2, group] = logit
    return v


@pytest.mark.parametrize('group,label', [(0, 'red_balloon'), (1, 'blue_balloon')])
def test_direct_scores_and_color(group, label):
    found = detect(values(3, group))
    assert len(found) == 1 and found[0]['accepted'] and found[0]['label'] == label
    assert found[0]['score'] == pytest.approx(1/(1+np.exp(-3)))
    assert sum(found[0]['scores']) == pytest.approx(1)


def test_subthreshold_proposal_not_promoted():
    found = detect(values(-1))
    assert len(found) == 1 and not found[0]['accepted'] and not found[0]['raw_accepted']


def test_invalid_threshold_and_nonfinite_output_rejected():
    for threshold in (0, 1, float('nan')):
        with pytest.raises(ValueError):
            detect(values(3), threshold)
    v = values(3)
    v[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        detect(v)


def test_nested_scores_below_crop_threshold_keep_real_scores():
    items = [dict(box=[0, 0, 100, 100], label='red_balloon', score=.6, accepted=True),
             dict(box=[20, 20, 40, 40], label='red_balloon', score=.7, accepted=True)]
    found = select_direct(items, .5)
    assert [r['accepted'] for r in found] == [False, True]
    assert [r['score'] for r in found] == [.6, .7]


def test_high_confidence_selection_matches_incumbent():
    items = [dict(box=[0, 0, 100, 100], label='red_balloon', score=.9, accepted=True),
             dict(box=[20, 20, 40, 40], label='red_balloon', score=.95, accepted=True)]
    assert select_direct(items, .5) == incumbent_select(items)
