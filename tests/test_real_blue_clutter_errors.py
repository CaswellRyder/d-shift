from copy import deepcopy

import pytest

from scripts.audit_real_blue_clutter_errors import attribute
from scripts.evaluate_red_blue_development import detection_counts, metrics


def fixture():
    target = dict(box=[0, 0, 10, 10], label='blue_balloon')
    detections = [dict(target, score=.9, accepted=True),
                  dict(target, score=.85, accepted=True),
                  dict(box=[30, 30, 40, 40], label='blue_balloon', score=.9, accepted=True)]
    frame = dict(id=0, source='test', frame_sha256='test', truth=[target], detections=detections)
    frame['counts'] = detection_counts(frame['truth'], detections)
    return dict(frames=[frame], metrics=metrics(frame['counts']))


def test_attribution_accounts_for_duplicate_and_nonoverlapping_false_alarms():
    result = fixture()
    report = attribute(result, result['frames'], result['frames'])
    assert sum(report['false_positive_counts'].values()) == 2
    assert report['false_negative_counts'] == {}
    assert report['geometric_target_coverage']['blue_balloon']['covered'] == 1


@pytest.mark.parametrize('field', ['id', 'source', 'frame_sha256', 'truth'])
def test_paired_frame_identity_required(field):
    result = fixture()
    parent = deepcopy(result['frames'])
    parent[0][field] = 'different'
    with pytest.raises(ValueError, match='Frame identity'):
        attribute(result, result['frames'], parent)


def test_different_proposal_geometry_rejected():
    result = fixture()
    parent = deepcopy(result['frames'])
    parent[0]['detections'][0]['box'] = [1, 1, 10, 10]
    with pytest.raises(ValueError, match='proposal geometry'):
        attribute(result, result['frames'], parent)
