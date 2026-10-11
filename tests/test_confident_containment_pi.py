from copy import deepcopy

import pytest

from scripts.audit_confident_containment_pi import validate_row
from scripts.build_confident_containment_pi import frozen_source
from scripts.compare_balloon_containment import confidence_ordered


def test_extracted_policy_matches_original_and_requires_no_research_imports():
    namespace = {}
    exec(compile(frozen_source(), '<frozen-containment-test>', 'exec'), namespace)
    rows = [dict(box=[0, 0, 100, 100], label='blue_balloon', score=.9, accepted=True),
            dict(box=[10, 10, 40, 40], label='blue_balloon', score=.99, accepted=True)]
    assert namespace['confidence_ordered'](rows) == confidence_ordered(rows)


def fixture():
    source = dict(panel='training', id=0, source='train/example', frame_sha256='abc', truth=[],
                  expected=dict(parent=[], confidence=[]))
    row = dict({k: source[k] for k in ('panel', 'id', 'source', 'frame_sha256', 'truth')},
        detections=[], neural_calls=0, localizer_calls=1, score_delta=0., box_delta=0.,
        processing_ms=10., search_ms=4., localizer_ms=3., classifier_ms=2., selection_ms=1., containment_extra_ms=.2)
    return row, source


def test_pi_row_recounts_empty_scene_and_validates_timings():
    row, source = fixture()
    assert validate_row(row, source, 'parent')['blue_balloon'] == dict(tp=0, fp=0, fn=0)


@pytest.mark.parametrize('key,value', [('processing_ms', 11.), ('containment_extra_ms', 2.),
    ('neural_calls', 1), ('localizer_calls', 2), ('source', 'other'), ('score_delta', .1)])
def test_pi_row_rejects_identity_workload_and_timing_drift(key, value):
    row, source = fixture()
    changed = deepcopy(row)
    changed[key] = value
    with pytest.raises(ValueError):
        validate_row(changed, source, 'parent')
