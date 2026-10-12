import copy

import pytest

from scripts.audit_proposal_quality_localizer import eligible
from scripts.proposal_quality_features import gate
from scripts.summarize_proposal_quality_localizer import verify_gate


def test_gate_recount_rejects_changed_classification_or_work():
    rows = [dict(proposal_origin='mser', accepted=True), dict(proposal_origin='learned', accepted=True)]
    parent = dict(id=1, source='train/example', truth=[], frame_sha256='same', detections=rows, neural_calls=1)
    quality = dict(parent, detections=gate(rows, [None, .3]), full_frame_neural_calls=1, quality_head_calls=1, quality_rows=1)
    verify_gate(parent, quality)
    changed = copy.deepcopy(quality)
    changed['detections'][0]['accepted'] = False
    with pytest.raises(ValueError):
        verify_gate(parent, changed)
    changed = copy.deepcopy(quality)
    changed['quality_head_calls'] = 0
    with pytest.raises(ValueError):
        verify_gate(parent, changed)


def test_gate_requires_both_seeds_and_target_retention():
    effect = dict(lost_targets=[], count_delta={'red_balloon': {'fp': -1}, 'blue_balloon': {'fp': 0}})
    panels = {'original': {f'quality{seed}_frozen': dict(quality_effect=copy.deepcopy(effect)) for seed in (47, 48)}}
    assert eligible(panels, 'frozen')
    panels['original']['quality48_frozen']['quality_effect']['lost_targets'] = ['target']
    assert not eligible(panels, 'frozen')
