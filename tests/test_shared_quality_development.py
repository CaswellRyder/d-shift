import copy

import pytest

from scripts.shared_proposal_quality import gate
from scripts.summarize_shared_quality_development import verify_frame


def fixture():
    row = dict(label='blue_balloon', score=.9, proposal_origin='mser', accepted=True)
    parent = dict(id=0, source='IMG_example', frame_sha256='same', truth=[], vertical_flip=True,
                  detections=[row], neural_calls=1)
    result = dict(parent, detections=gate([row], [.3]), full_frame_neural_calls=1, quality_head_calls=1, quality_rows=1)
    return parent, result


def test_development_recount_binds_orientation_and_work():
    parent, result = fixture()
    verify_frame(parent, result)
    for key, value in [('vertical_flip', False), ('quality_rows', 0), ('neural_calls', 2)]:
        changed = copy.deepcopy(result)
        changed[key] = value
        with pytest.raises(ValueError):
            verify_frame(parent, changed)


def test_development_recount_rejects_unrecorded_threshold_change():
    parent, result = fixture()
    changed = copy.deepcopy(result)
    changed['detections'][0]['box_quality'] = .9
    with pytest.raises(ValueError):
        verify_frame(parent, changed)
