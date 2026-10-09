import numpy as np
import pytest

from dtr.data import read_json
from scripts.train_balloon_box_refinement import check_config, decoded_box, partition, supervision
from scripts.train_balloon_quality import fold_ids
from scripts.evaluate_balloon_box_refinement import BoxPredictor, observations
from dtr.data import write_json
from scripts.compare_box_selection import locked_selection


@pytest.fixture
def config():
    return read_json('configs/balloon-box-refinement-20261008.json')


def row():
    return dict(kind='reviewed_balloon_overlap', box=[10, 10, 30, 30], crop_box=[0, 0, 40, 40],
                target_boxes=[[12, 14, 32, 34]])


def test_normalized_supervision_roundtrip(config):
    target, reason = supervision(row(), config)
    assert reason == 'admitted'
    box, applied = decoded_box(target['normalized_target'], row(), config)
    np.testing.assert_allclose(box, row()['target_boxes'][0])
    assert applied


@pytest.mark.parametrize('change,reason', [
    ({'kind':'exact_reviewed_background'},'negative'),
    ({'target_boxes':[]},'low_overlap'),
    ({'target_boxes':[[30,30,40,40]]},'low_overlap'),
    ({'target_boxes':[[10,10,30,30],[10,10,30,30]]},'ambiguous'),
    ({'target_boxes':[[10,10,41,35]]},'outside_crop')])
def test_unknown_ambiguous_or_invisible_targets_not_fitted(config, change, reason):
    assert supervision(dict(row(), **change), config) == (None, reason)


@pytest.mark.parametrize('values', [[.8,.8,.2,.2], [.5,.5,.51,.51], [0,0,.15,.15]])
def test_unsafe_or_disjoint_corrections_fallback(config, values):
    assert decoded_box(values,row(),config) == (row()['box'], False)


def test_output_clipped_to_visible_crop(config):
    box, applied = decoded_box([-.1,-.2,1.2,1.3],row(),config)
    assert box == [0,0,40,40] and applied


@pytest.mark.parametrize('values', [[0,0,float('nan'),1], [0,0,1], [0,0,float('inf'),1]])
def test_invalid_output_rejected(config, values):
    with pytest.raises(ValueError, match='output'):
        decoded_box(values,row(),config)


@pytest.mark.parametrize('key,value', [('deployment_approved',True),('training_approved',False),
    ('ridge_penalty',0),('holdout_fold',True),('minimum_match_margin',0),('minimum_decoded_side',0)])
def test_unplanned_configuration_rejected(config,key,value):
    config[key] = value
    with pytest.raises(ValueError):
        check_config(config)


def test_source_groups_and_source_hashes_never_cross_holdout(config):
    rows = [dict(row(),source_group=str(i),source_image=str(i)+'.jpg',source_sha256=str(i)) for i in range(30)]
    train,hold,excluded=partition(rows,config)
    assert not excluded and len(train)+len(hold)==30
    assert all(fold_ids(hold)==0) and all(fold_ids(train)!=0)
    train[0]['source_sha256']=hold[0]['source_sha256']
    with pytest.raises(ValueError,match='leakage'):
        partition(train+hold,config)
    with pytest.raises(ValueError,match='held-out'):
        partition(hold,config)


class FakeBoxPredictor:
    def __init__(self,config):
        self.metadata=dict(classes=['background','red_balloon','blue_balloon'],config=config)

    def predict(self,rgb):
        return np.array([.01,.98,.01]),np.array([.3,.35,.8,.85])


def old_prediction():
    return dict(row(),label='red_balloon',scores=[.01,.98,.01],score=.98,raw_accepted=True,
                accepted=False,suppressed=True,suppression_reason='confirmed_balloon_part',suppressed_by=2)


def test_only_geometry_changes_and_stale_suppression_removed(config):
    old=old_prediction()
    original,refined,delta=observations(FakeBoxPredictor(config),np.zeros((240,320,3),np.uint8),[old])
    assert delta==0 and original[0]['box']==old['box']
    np.testing.assert_allclose(refined[0]['box'],[12,14,32,34])
    assert refined[0]['proposal_box']==old['box'] and refined[0]['box_correction_applied']
    assert refined[0]['accepted'] and original[0]['accepted']
    assert original[0]['scores']==refined[0]['scores']
    assert 'suppressed_by' not in refined[0] and 'suppression_reason' not in original[0]
    assert not old['accepted']


def test_changed_classifier_or_saved_scores_rejected(config):
    old=old_prediction()
    with pytest.raises(ValueError,match='Frozen'):
        observations(FakeBoxPredictor(config),np.zeros((240,320,3),np.uint8),[dict(old,raw_accepted=False)])
    with pytest.raises(ValueError,match='saved class'):
        observations(FakeBoxPredictor(config),np.zeros((240,320,3),np.uint8),[dict(old,scores=[float('nan'),0,1])])


def test_wrong_model_contract_rejected_before_tensorflow(tmp_path):
    path=tmp_path/'wrong.tflite'
    path.write_bytes(b'wrong model')
    write_json(path.with_suffix('.json'),dict(kind='balloon_quality_research'))
    with pytest.raises(ValueError,match='identity'):
        BoxPredictor(path)


@pytest.mark.parametrize('rgb',[np.zeros((3,3),np.uint8),np.zeros((0,3,3),np.uint8),np.zeros((3,3,3),np.float32)])
def test_predictor_rejects_invalid_rgb(rgb):
    predictor=BoxPredictor.__new__(BoxPredictor)
    with pytest.raises(ValueError,match='RGB'):
        predictor.predict(rgb)


def test_locked_selection_never_resurrects_suppressed_parts():
    old=old_prediction()
    new=dict(old,accepted=True,suppressed=False,box=[12,14,32,34],proposal_box=old['box'],box_correction_applied=True)
    found=locked_selection([old],[new])[0]
    assert not found['accepted'] and found['suppressed']
    assert found['box']==[12,14,32,34] and old['box']==[10,10,30,30]


@pytest.mark.parametrize('field,value',[('proposal_box',[0,0,10,10]),('scores',[0,1,0]),
    ('box',[0,0,float('nan'),1]),('raw_accepted',False),('crop_box',[1,1,39,39])])
def test_locked_selection_rejects_mismatched_evidence(field,value):
    old=old_prediction()
    new=dict(old,proposal_box=old['box'],box_correction_applied=True)
    new[field]=value
    with pytest.raises(ValueError):
        locked_selection([old],[new])
