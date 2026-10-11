import copy

import numpy as np
import pytest

from scripts.window1_trial_data import CHANGED, matched_inputs, recovered_targets


def fixture():
    a = dict(x=np.zeros((2547, 1, 1, 3), np.float32), p={}, scene_ids=[])
    for key in ('y', 'preserve', 'scene_mask', 'weights', 'probes', 'probe_labels', 'batch_labels'):
        a[key] = np.zeros(3)
    b = copy.deepcopy(a)
    b['x'][CHANGED] = 1
    return a, b


def test_eight_expected_replacements_only():
    a, b = fixture()
    assert matched_inputs(a, b) == CHANGED


@pytest.mark.parametrize('key', ['y', 'weights', 'scene_mask', 'batch_labels', 'x'])
def test_confounded_arms_rejected(key):
    a, b = fixture()
    b[key][0] += 1
    with pytest.raises((ValueError, AssertionError)):
        matched_inputs(a, b)


def scenes():
    return dict(frames=[dict(id=f, truth=[dict(label='blue_balloon', box=[0, 0, 10, 10], annotation_id=t)],
        detections=[dict(label='blue_balloon', accepted=True, score=.9, box=[0, 0, 10, 10])])
        for f, t in ((9, 23), (20, 50))])


def test_recovery_requires_both_actual_target_identities():
    assert recovered_targets(scenes())
    result = scenes()
    result['frames'][0]['truth'][0]['annotation_id'] = 999
    assert not recovered_targets(result)


@pytest.mark.parametrize('field,value', [('accepted', False), ('label', 'red_balloon'), ('box', [20, 20, 30, 30])])
def test_recovery_requires_accepted_correct_color_localization(field, value):
    result = scenes()
    result['frames'][1]['detections'][0][field] = value
    assert not recovered_targets(result)
