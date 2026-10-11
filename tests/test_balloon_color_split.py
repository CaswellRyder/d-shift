import cv2
import numpy as np
import pytest

from scripts.research_balloon_color_split import centers_for,color_split_candidates,proposals,tiny_nested
from scripts.research_balloon_search import experimental
from dtr.tracking import iou


def scene():
    rgb=np.full((240,320,3),160,np.uint8)
    cv2.rectangle(rgb,(10,130),(290,180),(25,20,125),-1)
    cv2.ellipse(rgb,(105,105),(44,33),0,0,360,(15,125,235),-1)
    cv2.ellipse(rgb,(225,104),(35,29),0,0,360,(25,90,205),-1)
    return rgb


def test_deterministic_and_no_input_mutation():
    rgb=scene()
    copy=rgb.copy()
    a=color_split_candidates(rgb)
    assert a==color_split_candidates(rgb) and np.array_equal(rgb,copy)
    assert a and len(a)<=12
    assert all(r["color_group"]==1 and r["proposal_origin"]=="lab_blue_split" for r in a)


def test_separates_touching_blue_tones_in_fixture_not_real_accuracy():
    boxes=color_split_candidates(scene())
    for truth in ([61,72,150,139],[190,75,261,134]):
        assert max(iou(truth,r["box"]) for r in boxes)>.7


def test_preserves_baseline_and_bounds_neural_calls():
    rgb=scene()
    original=experimental(rgb,"mser")
    combined=proposals(rgb)
    assert combined[:len(original)]==original
    assert len(combined)<=len(original)+4


def test_fragment_guard_preserves_substantial_and_different_color_boxes():
    parent=dict(box=[0,0,100,100],color_group=1)
    assert tiny_nested(dict(box=[10,10,20,20],color_group=1),[parent])
    assert not tiny_nested(dict(box=[0,0,50,50],color_group=1),[parent])
    assert not tiny_nested(dict(box=[10,10,20,20],color_group=0),[parent])
    assert not tiny_nested(dict(box=[95,95,105,105],color_group=1),[parent])


def test_guarded_search_preserves_baseline_and_bounds_work():
    rgb=scene()
    base=experimental(rgb,"mser")
    guarded=proposals(rgb,fragment_guard=True)
    assert guarded[:len(base)]==base
    assert len(guarded)<=len(base)+4
    assert not any(tiny_nested(r,base) for r in guarded[len(base):])


def test_flat_color_cannot_create_split():
    assert color_split_candidates(np.full((240,320,3),(10,20,200),dtype=np.uint8))==[]
    assert color_split_candidates(np.zeros((240,320,3),np.uint8))==[]
    assert centers_for(np.ones((100,3),np.float32)) is None


@pytest.mark.parametrize("array",[np.zeros((32,32,3),np.uint8),np.zeros((240,320,3),np.float32)])
def test_input_contract(array):
    with pytest.raises(ValueError):
        color_split_candidates(array)
