import numpy as np
import pytest

from scripts.diagnose_clutter_stage_attribution import crops, difference
from scripts.diagnose_clutter_round3_feature import replay


def frame():
    return dict(id=0,source='train/a',frame_sha256='abc',truth=[dict(label='red_balloon',box=[10,10,20,20])],
                detections=[dict(box=[10,10,20,20],crop_box=[8,8,22,22])])


def test_crop_identity_and_geometry_guards():
    f=frame()
    rgb=np.zeros((240,320,3),np.uint8)
    pixels={'train/a':('abc',f['truth'],rgb)}
    assert crops([f],pixels).shape==(1,64,64,3)
    with pytest.raises(ValueError):
        crops([f],{'train/a':('wrong',f['truth'],rgb)})
    with pytest.raises(ValueError):
        crops([f],{})


def test_stage_attribution_exposes_target_loss_not_only_aggregate_accuracy():
    f=frame()
    parent=replay([f],np.array([[.01,.98,.01]]))
    after=replay([f],np.array([[.98,.01,.01]]))
    result=difference(after,parent)
    assert result['count_delta']['red_balloon']==dict(tp=-1,fp=0,fn=1)
    assert len(result['lost_targets'])==1
    assert not result['gained_targets']
