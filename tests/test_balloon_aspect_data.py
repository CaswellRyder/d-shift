import copy

import numpy as np
import pytest

from scripts.balloon_aspect_data import matched_inputs
from scripts.audit_balloon_aspect import training_gate


def pair():
    a = {k:np.zeros(3) for k in ('y','weights','preserve','scene_mask','probe_labels','batch_labels')}
    a.update(x=np.zeros((3,2,2,3)),probes=np.zeros((2,2,2,3)),batch_pixels=np.zeros((2,2,2,3)),
             sample_identity=[1,2,3],scene_ids=[1],p={'epochs':80,'recipe':'original'})
    b = copy.deepcopy(a)
    for k in ('x','probes','batch_pixels'):
        b['teacher_'+k] = a[k].copy()
        b[k][:] = 127
    b['p']['recipe'] = 'aspect-preserving'
    return a,b


def test_control_teacher_and_supervision_match():
    a,b = pair()
    assert matched_inputs(a,b) == [0,1,2]


@pytest.mark.parametrize('change',['teacher_x','y','weights','sample_identity','epochs'])
def test_changed_control_or_supervision_rejected(change):
    a,b = pair()
    if change == 'sample_identity':
        b[change][0] = 5
    elif change == 'epochs':
        b['p'][change] = 81
    else:
        b[change][0] = 1
    with pytest.raises((ValueError,AssertionError)):
        matched_inputs(a,b)


def test_gate_cannot_trade_lost_targets_for_false_positive_reduction():
    e = dict(lost_targets=[],gained_targets=[],count_delta={'blue':dict(tp=0,fp=-1,fn=0)})
    assert training_gate(e,e,[7,2])
    bad = copy.deepcopy(e)
    bad['lost_targets'] = ['target']
    assert not training_gate(e,bad,[7,2])
