import copy

import pytest

from scripts import evaluate_spatial_balloon_box as module
from scripts.evaluate_red_blue_development import metrics


@pytest.fixture
def source(monkeypatch):
    monkeypatch.setattr(module,'sha256',str)
    counts={k:dict(tp=0,fp=0,fn=0) for k in ('red_balloon','blue_balloon')}
    frame=dict(id=0,source='frame',frame_sha256='pixels',truth=[],detections=[],counts=counts)
    report=dict(model_sha256='model',queue_sha256='queue',review_sha256='review',threshold=.8,
                matching_iou=.5,selection='split_merged',split_max_aspect=2.42913421076193,
                test_evaluated=False,deployment_approved=False,
                variants=dict(baseline=dict(frames=[frame],metrics=metrics(counts))))
    ref=dict(id=0,source='frame',sha256='pixels',provisional_truth=[])
    return report,{0:(ref,None)},dict(queue_sha256='queue'),'review',dict(source_tflite_sha256='model')


def test_verified_parent_report_accepted_without_mutation(source):
    before=copy.deepcopy(source)
    assert module.checked_source(*source)==source[0]['variants']['baseline']
    assert source==before


@pytest.mark.parametrize('fault',['model','threshold','scope','selection','coverage','truth','pixels','counts'])
def test_mismatched_parent_evidence_rejected(source,fault):
    report=source[0]
    if fault=='model':
        report['model_sha256']='wrong'
    elif fault=='threshold':
        report['threshold']=.7
    elif fault=='scope':
        report['deployment_approved']=True
    elif fault=='selection':
        report['selection']='parts'
    else:
        baseline=report['variants']['baseline']
        if fault=='coverage':
            baseline['frames']=[]
        elif fault=='truth':
            source[1][0][0]['provisional_truth']=[dict(label='blue_balloon',box=[0,0,10,10])]
        elif fault=='pixels':
            baseline['frames'][0]['frame_sha256']='changed'
        else:
            baseline['frames'][0]['counts']['blue_balloon']['fp']=1
    with pytest.raises(ValueError):
        module.checked_source(*source)
