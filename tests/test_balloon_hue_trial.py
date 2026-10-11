import copy

import numpy as np
import pytest

from scripts.balloon_hue_trial_data import START, STOP, matched_inputs
from scripts.balloon_hue_augmentation import recolor
from scripts.audit_balloon_hue import training_gate


def pair():
    a = dict(x=np.zeros((2675, 1, 1, 3), np.float32), y=np.ones(2675, int),
             p={'epochs':80}, scene_ids=[])
    a['x'][START:STOP] = [200, 40, 40]
    for k in ('preserve','scene_mask','weights','probes','probe_labels','batch_pixels','batch_labels'):
        a[k] = np.zeros(3)
    b = copy.deepcopy(a)
    for i in range(START, STOP):
        b['x'][i], b['y'][i], _ = recolor(a['x'][i], int(a['y'][i]))
    return a, b


def test_only_rendered_pixels_and_semantic_color_labels_change():
    a, b = pair()
    assert matched_inputs(a,b) == list(range(START, STOP))


@pytest.mark.parametrize('change', ['outside','label','render','weight','schedule'])
def test_mismatched_intervention_rejected(change):
    a, b = pair()
    if change == 'outside':
        b['x'][0] = 1
    elif change == 'label':
        b['y'][START] = 0
    elif change == 'render':
        b['x'][START] = 1
    elif change == 'weight':
        b['weights'][0] = 1
    else:
        b['p']['epochs'] = 81
    with pytest.raises((ValueError, AssertionError)):
        matched_inputs(a,b)


def effect(tp=0, fp=0, lost=False):
    return dict(count_delta={'red_balloon':dict(tp=tp,fp=fp,fn=-tp),
                             'blue_balloon':dict(tp=0,fp=0,fn=0)},
                gained_targets=['target'] if tp>0 else [], lost_targets=['target'] if lost else [])


def test_improvement_required_with_all_retention_gates():
    assert training_gate(effect(fp=-1),effect(),[7,2])
    assert training_gate(effect(),effect(tp=1),[7,2])
    assert not training_gate(effect(),effect(),[7,2])
    assert not training_gate(effect(fp=-1,lost=True),effect(),[7,2])
    assert not training_gate(effect(fp=-1),effect(fp=1),[7,2])
    assert not training_gate(effect(fp=-1),effect(),[7,8])
