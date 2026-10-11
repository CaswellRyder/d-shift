import numpy as np
import pytest

from scripts.balloon_dense_public_data import encode_public, expected


def test_unknown_public_regions_remain_masked():
    truth=[dict(label='blue_balloon',box=[32,32,48,48])]
    target,accepted,skipped,collisions=encode_public(truth,[[96,96,128,128]])
    assert not skipped and not collisions and accepted==[[96,96,128,128]]
    assert np.count_nonzero(target[:,:,2:4])==10
    assert not target[0,0,2:4].any()
    assert target[2,2,1]==1 and target[2,2,8]==1


def test_conflicting_background_is_skipped_not_used_to_erase_positive():
    truth=[dict(label='red_balloon',box=[32,32,48,48])]
    target,accepted,skipped,_=encode_public(truth,[[0,0,64,64]])
    assert accepted==[] and skipped==[[0,0,64,64]]
    assert target[2,2,0]==1 and target[2,2,8]==1


def test_public_encoder_does_not_silently_accept_invalid_geometry():
    with pytest.raises(ValueError):
        encode_public([],[[30,0,10,20]])
    with pytest.raises(ValueError):
        expected('test')
