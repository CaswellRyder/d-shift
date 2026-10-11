import cv2
import numpy as np
import pytest

from dtr.tracking import iou
from scripts.research_balloon_edge_ellipse import edge_candidates,fit_candidate,proposals
from scripts.evaluate_balloon_edge_search import coverage
from scripts.research_balloon_search import experimental


def fixture():
    rgb=np.full((240,320,3),180,np.uint8)
    cv2.rectangle(rgb,(20,126),(285,151),(20,25,125),-1)
    cv2.ellipse(rgb,(112,100),(38,30),0,0,360,(20,95,225),-1)
    return rgb


def test_luminance_boundary_can_separate_synthetic_touching_blue_shapes():
    candidates=edge_candidates(fixture())
    assert max((iou(r['box'],[74,70,151,131]) for r in candidates),default=0)>.7


def test_deterministic_no_mutation_and_bounded_added_calls():
    rgb=fixture()
    before=rgb.copy()
    found=proposals(rgb)
    baseline=experimental(rgb,'mser')
    assert found==proposals(rgb) and found[:len(baseline)]==baseline
    assert len(found)<=min(12,len(baseline)+2)
    np.testing.assert_array_equal(rgb,before)


def test_nonblue_and_flat_images_do_not_create_balloon_proposals():
    assert edge_candidates(np.full((240,320,3),80,np.uint8))==[]
    assert edge_candidates(fixture()[:,:,::-1].copy())==[]


def test_degenerate_points_fail_closed():
    assert fit_candidate(np.zeros((20,1,2),np.float32),np.ones((240,320),bool)) is None
    assert fit_candidate(np.zeros((4,1,2),np.float32),np.ones((240,320),bool)) is None


@pytest.mark.parametrize('rgb',[np.zeros((240,320,3),np.float32),np.zeros((24,32,3),np.uint8)])
def test_explicit_frame_contract(rgb):
    with pytest.raises(ValueError):
        edge_candidates(rgb)


def test_training_coverage_is_one_to_one_and_color_specific():
    truth=[dict(label='blue_balloon',box=[10,10,30,30])]*2
    candidates=[dict(color_group=1,box=[10,10,30,30])]
    assert coverage(truth,candidates)['blue_balloon']==dict(targets=2,localized=1)
    assert coverage(truth,[dict(color_group=0,box=[10,10,30,30])])['blue_balloon']['localized']==0


def test_ellipse_fit_work_is_bounded(monkeypatch):
    from scripts import research_balloon_edge_ellipse as module
    contour=np.zeros((500,1,2),np.int32)
    calls=[]
    monkeypatch.setattr(module.cv2,'findContours',lambda *args:([contour]*100,None))
    monkeypatch.setattr(module,'fit_candidate',lambda points,blue:calls.append(len(points)))
    assert module.edge_candidates(fixture())==[]
    assert len(calls)==24*13 and max(calls)<=256
