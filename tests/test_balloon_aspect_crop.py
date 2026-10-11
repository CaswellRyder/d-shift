import numpy as np
import pytest

from scripts.balloon_aspect_crop import resized


@pytest.mark.parametrize('shape,region', [((10,20),(slice(16,48),slice(0,64))),((20,10),(slice(0,64),slice(16,48)))])
def test_aspect_is_preserved_with_centered_neutral_padding(shape,region):
    x = np.full((*shape,3),200,np.uint8)
    y = resized(x)
    assert y.shape == (64,64,3) and y.dtype == np.float32
    expected = np.full((64,64,3),127,np.float32)
    expected[region] = 200
    np.testing.assert_array_equal(y,expected)


def test_square_inputs_identical_to_original_resize():
    x = np.arange(12*12*3,dtype=np.uint8).reshape(12,12,3)
    np.testing.assert_array_equal(resized(x),resized(x,False))


def test_extreme_aspect_keeps_one_pixel_minimum():
    x = np.full((1,1000,3),200,np.uint8)
    y = resized(x)
    assert np.count_nonzero(y[...,0]==200) == 64


@pytest.mark.parametrize('x',[np.zeros((0,1,3),np.uint8),np.zeros((2,2),np.uint8),np.zeros((2,2,3),np.float32)])
def test_invalid_input_rejected(x):
    with pytest.raises(ValueError):
        resized(x)


def test_existing_flip_applied_after_resize(tmp_path):
    from PIL import Image
    from scripts.balloon_aspect_data import crop
    x = np.arange(7*11*3,dtype=np.uint8).reshape(7,11,3)
    path = tmp_path/'crop.png'
    Image.fromarray(x).save(path)
    row = dict(file=str(path),split='train',source_image='train/example',view='horizontal_flip')
    for aspect in (True,False):
        np.testing.assert_array_equal(crop(row,aspect),resized(x,aspect)[:,::-1])
