import copy

import pytest

from scripts.audit_shared_quality import eligible
from scripts.shared_proposal_quality import gate
from scripts.summarize_shared_quality import verify_gate, meets94


def test_exact_gate_and_work_recount():
    row = dict(label='red_balloon', score=.9, proposal_origin='mser', accepted=True)
    parent = dict(id=0, source='train/source', frame_sha256='same', truth=[], detections=[row], neural_calls=1)
    result = dict(parent, detections=gate([row], [.3]), full_frame_neural_calls=1, quality_head_calls=1, quality_rows=1)
    verify_gate(parent, result)
    bad = copy.deepcopy(result)
    bad['detections'][0]['score'] = .99
    with pytest.raises(ValueError):
        verify_gate(parent, bad)
    bad = copy.deepcopy(result)
    bad['quality_rows'] = 0
    with pytest.raises(ValueError):
        verify_gate(parent, bad)


def test_qualification_fraction_not_rounded_display():
    assert meets94({'red': dict(precision=.94, recall=.95), 'blue': dict(precision=.95, recall=.94)})
    assert not meets94({'red': dict(precision=.93999, recall=.99)})


def test_paired_gate_rejects_a_single_seed_regression():
    effect = dict(lost_targets=[], count_delta={'red_balloon': {'fp': -1}, 'blue_balloon': {'fp': 0}})
    panels = {'original': {f'iou{s}_frozen': dict(quality_effect=copy.deepcopy(effect)) for s in (47, 48)}}
    assert eligible(panels, 'iou', 'frozen')
    panels['original']['iou48_frozen']['quality_effect']['lost_targets'] = ['target']
    assert not eligible(panels, 'iou', 'frozen')
