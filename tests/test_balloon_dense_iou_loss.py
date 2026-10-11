import numpy as np
import pytest

from scripts.balloon_dense_iou_loss import generalized_iou,geometry_boxes,overlap_penalty,giou_loss
from scripts.prepare_balloon_dense_proposals import encode
from scripts.train_balloon_dense_proposals import sparse_loss


def test_giou_exact_overlap_disjoint_and_degenerate():
    import tensorflow as tf
    a = tf.constant([[0.,0.,1.,1.]]*3+[[0.,0.,0.,0.]])
    b = tf.constant([[0.,0.,1.,1.],[0.,0.,.5,.5],[2.,0.,3.,1.],[0.,0.,0.,0.]])
    np.testing.assert_allclose(generalized_iou(a,b),[1.,.25,-1/3,0.],atol=1e-6)


def test_geometry_decodes_original_pixel_coordinates_and_clips_edges():
    import tensorflow as tf
    geometry = np.zeros((1,15,20,4),np.float32)
    geometry[0,5,5] = [.5,.5,.2,.2]
    actual = geometry_boxes(tf.constant(geometry)).numpy()
    np.testing.assert_allclose(actual[0,5,5]*[320,240,320,240],[56,64,120,112],atol=1e-5)
    geometry[0,0,0] = [0.,0.,.5,.5]
    np.testing.assert_allclose(geometry_boxes(tf.constant(geometry))[0,0,0],[0,0,.25,.25])


def example():
    target,_ = encode([dict(label='blue_balloon',box=[64,64,96,96])],[])
    prediction = np.zeros((1,15,20,6),np.float32)
    prediction[...,2:] = target[None,...,4:8]
    return target[None],prediction


def test_perfect_geometry_has_zero_extra_loss():
    import tensorflow as tf
    target,prediction = map(tf.constant,example())
    assert float(overlap_penalty(target,prediction))==pytest.approx(0.,abs=1e-6)
    assert float(giou_loss(target,prediction))==pytest.approx(float(sparse_loss(target,prediction)))


def test_overlap_penalty_has_center_only_finite_gradient():
    import tensorflow as tf
    target,prediction = example()
    prediction[...,2:] = [.5,.5,.4,.4]
    variable = tf.Variable(prediction)
    with tf.GradientTape() as tape:
        loss = overlap_penalty(tf.constant(target),variable)
    gradient = tape.gradient(loss,variable).numpy()
    assert np.isfinite(gradient).all() and float(loss)>0
    np.testing.assert_array_equal(gradient[0,0,0],np.zeros(6))
    np.testing.assert_array_equal(gradient[...,:2],np.zeros((1,15,20,2)))
    assert np.any(gradient[0,5,5,2:]!=0)


def test_no_centers_means_no_geometry_penalty_or_gradient():
    import tensorflow as tf
    target = tf.zeros((1,15,20,9))
    prediction = tf.Variable(tf.ones((1,15,20,6))*.5)
    with tf.GradientTape() as tape:
        loss = overlap_penalty(target,prediction)
    assert float(loss)==0
    np.testing.assert_array_equal(tape.gradient(loss,prediction),np.zeros((1,15,20,6)))


def test_matched_trial_allows_only_objective_difference():
    from scripts.summarize_dense_overlap_trial import matched
    common = dict(manifest_sha256='data',script_sha256='script',preparation_script_sha256='prep',
        seed=47,epochs=80,batch_size=8,learning_rate=.001,parameters=2342,fixed_final_epoch=True,
        folds=[1,0],groups=['a','b'],complete_background={'guard_pixels':16},loss_helper_sha256='loss')
    control = dict(common,box_objective='l1',giou_weight=0)
    treatment = dict(common,box_objective='l1_giou',giou_weight=2)
    matched(control,treatment)
    with pytest.raises(ValueError,match='epochs'):
        matched(control,dict(treatment,epochs=100))
    with pytest.raises(ValueError,match='weight'):
        matched(control,dict(treatment,giou_weight=3))
