from copy import deepcopy

import numpy as np
import pytest
from PIL import Image

from scripts.research_blue_mirror import mirror_crop, verify_blue


class FakePredictor:
    metadata = dict(size=64, threshold=.8, classes=['background','red_balloon','blue_balloon'])

    def __init__(self, label='blue_balloon', accepted=True):
        self.calls = []
        self.prediction = dict(label=label, accepted=accepted, score=.9, scores=[.05,.05,.9])

    def predict(self, crop):
        self.calls.append(crop.copy())
        return deepcopy(self.prediction)


def detection(label='blue_balloon', accepted=True):
    return dict(label=label, accepted=accepted, crop_box=[2,3,20,21], box=[4,5,18,19],
        score=.97, scores=[.01,.02,.97], raw_accepted=True, suppressed=not accepted)


def test_mirror_is_after_canonical_resize_and_does_not_mutate():
    rgb = np.random.default_rng(47).integers(0,256,(24,32,3),dtype=np.uint8)
    before = rgb.copy()
    expected = np.asarray(Image.fromarray(rgb[3:21,2:20]).resize((64,64),Image.Resampling.BILINEAR))
    actual = mirror_crop(rgb,[2,3,20,21])
    np.testing.assert_array_equal(actual,expected[:,::-1])
    np.testing.assert_array_equal(rgb,before)
    assert actual.flags.c_contiguous


@pytest.mark.parametrize('box', [[0,0,33,24],[0,0,0,1],[-1,0,1,1],[0.,0,2,2]])
def test_invalid_crop_rejected(box):
    with pytest.raises(ValueError):
        mirror_crop(np.zeros((24,32,3),dtype=np.uint8),box)


@pytest.mark.parametrize('label,accepted,keep', [('blue_balloon',True,True),
    ('blue_balloon',False,False),('red_balloon',True,False),('background',False,False)])
def test_only_selected_blue_is_checked_no_resurrection(label,accepted,keep):
    original = [detection(),detection(accepted=False),detection('red_balloon'),detection('background',False)]
    saved = deepcopy(original)
    predictor = FakePredictor(label,accepted)
    result = verify_blue(predictor,np.zeros((24,32,3),dtype=np.uint8),original)
    assert len(predictor.calls) == 1
    assert original == saved
    assert result[1:] == original[1:]
    assert result[0]['accepted'] is keep
    assert result[0]['blue_mirror']['retained'] is keep
    assert all(result[0][k] == original[0][k] for k in original[0] if k != 'accepted')
    with pytest.raises(ValueError,match='twice'):
        verify_blue(predictor,np.zeros((24,32,3),dtype=np.uint8),result)


def test_nonfinite_mirror_fails_closed():
    predictor = FakePredictor()
    predictor.prediction['scores'][2] = float('nan')
    with pytest.raises(ValueError,match='Nonfinite'):
        verify_blue(predictor,np.zeros((24,32,3),dtype=np.uint8),[detection()])


def test_protocol_change_rejected():
    predictor = FakePredictor()
    predictor.metadata = dict(predictor.metadata,threshold=.7)
    with pytest.raises(ValueError,match='operating point'):
        verify_blue(predictor,np.zeros((24,32,3),dtype=np.uint8),[detection()])
