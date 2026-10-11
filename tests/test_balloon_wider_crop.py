import pytest

from scripts.balloon_wider_crop import wider_crop


@pytest.mark.parametrize('box',[[0,0,5,5],[315,235,320,240],[20,30,33,42],[0,0,320,240],[10,40,30,80]])
def test_context_contains_old_crop_and_stays_inside_image(box):
    a,b,c,d=wider_crop(box,(240,320,3))
    assert 0<=a<=box[0]<box[2]<=c<=320
    assert 0<=b<=box[1]<box[3]<=d<=240
    assert c-a>=16 and d-b>=16


def test_fixed_scale_and_invalid_inputs():
    assert wider_crop([10,40,30,80],(240,320,3))==[5,30,35,90]
    with pytest.raises(ValueError):
        wider_crop([0,0,0,5],(240,320,3))
    with pytest.raises(ValueError):
        wider_crop([-1,0,2,5],(240,320,3))
