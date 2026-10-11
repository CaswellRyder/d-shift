import numpy as np
import pytest

from scripts.prepare_balloon_dense_proposals import encode
from scripts.train_balloon_dense_proposals import network,sparse_loss


def test_unknown_scene_pixels_have_no_negative_supervision():
    target,collision = encode([dict(label='blue_balloon',box=[64,64,96,96])],[])
    assert collision == 0 and target.shape == (15,20,9)
    assert np.all(target[0,0] == 0)
    assert target[5,5,1] == 1 and target[5,5,8] == 1
    np.testing.assert_array_equal(target[5,5,2:4],[1,1])


def test_background_requires_a_whole_grid_cell_and_no_balloon_overlap():
    target,_ = encode([],[[1,1,15,15]])
    assert not target.any()
    target,_ = encode([],[[0,0,16,16]])
    np.testing.assert_array_equal(target[0,0,2:4],[1,1])
    assert not target[:,:,:2].any() and not target[:,:,8].any()
    with pytest.raises(ValueError,match='overlaps'):
        encode([dict(label='red_balloon',box=[0,0,16,16])],[[0,0,16,16]])


def test_colliding_centers_do_not_get_contradictory_box_targets():
    rows = [dict(label='red_balloon',box=[0,0,10,10]),dict(label='blue_balloon',box=[2,2,12,12])]
    target,collision = encode(rows,[])
    assert collision == 1 and target[0,0,8] == 0
    np.testing.assert_array_equal(target[0,0,:2],[1,1])


def test_unknown_logits_have_no_loss_gradient():
    import tensorflow as tf
    target,_ = encode([dict(label='blue_balloon',box=[64,64,96,96])],[])
    values = tf.Variable(tf.zeros((1,15,20,6)))
    with tf.GradientTape() as tape:
        loss = sparse_loss(tf.constant(target[None]),values)
    gradient = tape.gradient(loss,values).numpy()
    np.testing.assert_array_equal(gradient[0,0,0],np.zeros(6))
    assert np.any(gradient[0,5,5] != 0) and np.isfinite(gradient).all()


def test_localizer_has_explicit_small_dense_output_contract():
    model = network()
    assert model.input_shape == (None,120,160,3)
    assert model.output_shape == (None,15,20,6)
    assert model.count_params() < 4000


def test_decoder_and_supplement_keep_bounds_and_neural_call_limit():
    from scripts.audit_balloon_dense_proposals import decode,supplement
    values = np.zeros((15,20,6),np.float32)
    values[:,:,:2] = -20.
    values[:,:,2:] = [.5,.5,.2,.2]
    values[5,5,1] = 5.
    found = decode(values)
    assert len(found) == 1 and found[0]['color_group'] == 1
    np.testing.assert_allclose(found[0]['box'],[56,64,120,112],atol=1e-5)
    assert supplement(found,found) == found
    assert supplement(found*12,found) == found*12
    with pytest.raises(ValueError,match='finite'):
        decode(np.full((15,20,6),np.nan))
