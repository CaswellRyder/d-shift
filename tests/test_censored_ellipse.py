import cv2
import numpy as np
import pytest

from dtr.tracking import iou
from scripts.research_censored_ellipse import fit_box, proposals
from scripts.research_balloon_search import experimental


@pytest.mark.parametrize('center,axes,angle', [((160,120),(50,25),0), ((160,120),(25,50),30), ((8,120),(25,40),0)])
def test_free_center_fit_on_visible_and_clipped_ellipses(center, axes, angle):
    image = np.zeros((240,320), np.uint8)
    cv2.ellipse(image, center, axes, angle, 0,360,255,-1)
    mask = np.pad(image,1)
    distance = cv2.distanceTransform(mask,cv2.DIST_L2,5)
    y,x = np.unravel_index(distance.argmax(),distance.shape)
    box = fit_box(mask,x-1,y-1,float(distance[y,x]))
    points = cv2.findNonZero(image)
    a,b,w,h = cv2.boundingRect(points)
    assert box is not None
    assert iou(box,[a,b,a+w,b+h]) >= .8


def test_degenerate_fit_and_input_contract():
    assert fit_box(np.zeros((242,322),np.uint8),100,100,5) is None
    with pytest.raises(ValueError):
        proposals(np.zeros((240,320,3),np.float32))


def test_preserves_baseline_and_call_cap():
    rgb = np.zeros((240,320,3),np.uint8)
    cv2.ellipse(rgb,(150,120),(40,25),20,0,360,(0,0,230),-1)
    base = experimental(rgb,'mser')
    output = proposals(rgb)
    assert output[:len(base)] == base
    assert len(output) <= min(12,len(base)+2)
