import numpy as np
import pytest

from scripts.train_runtime_positives import calculate
from scripts.build_red_blue_proposal_positives import matched_target


def arrays():
    labels = np.arange(1824)%3
    logits = np.eye(3,dtype=np.float32)[labels]*10
    runtime_labels = np.array([1]*70+[2]*69)
    result = dict(before=logits.copy(),after=logits.copy(),labels=labels,ordinary=np.arange(1824)<1533,
        runtime_labels=runtime_labels,runtime_after=np.eye(3,dtype=np.float32)[runtime_labels]*10)
    for key,n,label in (('blue_after',23,2),('clutter_after',22,0),('real_after',39,0),('new_after',15,0)):
        result[key] = np.tile(np.eye(3,dtype=np.float32)[label]*10,(n,1))
    return result


def test_runtime_gate_requires_base_and_each_runtime_positive():
    a = arrays()
    assert calculate(a)['training_gate_pass']
    a['runtime_after'][0] = [0,0,10]
    assert not calculate(a)['training_gate_pass']
    a = arrays()
    a['after'][2] = [10,0,0]
    assert not calculate(a)['training_gate_pass']


def test_runtime_nonfinite_rejected():
    a = arrays()
    a['runtime_after'][0,0] = np.nan
    with pytest.raises(ValueError):
        calculate(a)


def test_positive_label_inheritance_is_not_classifier_pseudolabeling():
    c = dict(box=[20,20,60,60],crop_box=[15,15,65,65],color_group=1)
    target = dict(scan_box=[20,20,60,60],label='blue_balloon',annotation_id=0)
    assert matched_target(c,[target])==target
    assert matched_target(dict(c,color_group=0),[target]) is None
    assert matched_target(c,[target,dict(target,annotation_id=1)]) is None
    assert matched_target(c,[dict(target,scan_box=[50,50,90,90])]) is None
