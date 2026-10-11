import numpy as np
import pytest

from scripts.balloon_positive_protection import retention, feasible


def test_retention_tracks_individual_targets_not_net_counts():
    before = np.array([[0., 4., 0.], [0., 1., 0.], [0., 0., 4.]])
    after = np.array([[0., 1., 0.], [0., 4., 0.], [0., 0., 4.]])
    result = retention(before, after, [1, 1, 2], [True, True, False])
    assert result['red_balloon']['lost_indices'] == [0]
    assert result['red_balloon']['retained_fraction'] == 0.
    assert result['red_balloon']['current_argmax_correct'] == 2
    assert result['blue_balloon']['retained_fraction'] is None


def test_feasibility_cannot_pass_by_rejecting_everything():
    metrics = dict(retained_fraction=.99, current_argmax_correct=90, parent_argmax_correct=90)
    protocol = dict(minimum_parent_acceptance_retained=.99, minimum_reviewed_background_correct=11)
    report = dict(retention={'red_balloon': metrics, 'blue_balloon': metrics}, reviewed_background_correct=11)
    assert feasible(report, protocol)
    assert not feasible(dict(report, reviewed_background_correct=10), protocol)
    assert not feasible(dict(report, retention={'blue_balloon': dict(metrics, retained_fraction=0)}), protocol)
    assert not feasible(dict(report, retention={}), protocol)


def test_hinge_protects_only_ordinary_parent_accepted_correct_positives():
    tf = pytest.importorskip('tensorflow')
    from scripts.balloon_positive_protection import protected_loss
    from scripts.train_balloon_reviewed_clutter import loss_function
    labels = np.eye(3)[[1, 1, 0, 1, 1]].astype(np.float32)
    teacher = np.array([[0, 4, 0], [0, 4, 0], [4, 0, 0], [4, 0, 0], [0, 1, 0]], np.float32)
    targets = tf.constant(np.column_stack((labels, teacher, [1, 0, 1, 1, 1])), tf.float32)
    logits = tf.Variable(np.zeros((5, 3), np.float32))
    with tf.GradientTape() as tape:
        penalty = protected_loss()(targets, logits) - loss_function()(targets, logits)
        total = tf.reduce_sum(penalty)
    gradient = tape.gradient(total, logits).numpy()
    assert penalty.numpy()[0] > 0
    np.testing.assert_array_equal(penalty.numpy()[1:], np.zeros(4))
    assert gradient[0, 1] < 0 and gradient[0, 0] > 0 and gradient[0, 2] > 0
    np.testing.assert_allclose(protected_loss()(targets, logits + 50), protected_loss()(targets, logits), atol=.0002)
    stronger = tf.constant([[0., 5., 0.]] * 5)
    np.testing.assert_allclose((protected_loss()(targets, stronger) - loss_function()(targets, stronger)).numpy(), 0)


def test_low_confidence_correct_positive_is_outside_current_guard():
    tf = pytest.importorskip('tensorflow')
    from scripts.balloon_positive_protection import protected_loss
    from scripts.train_balloon_reviewed_clutter import loss_function
    # This boundary explains the observed residual forgetting; changing it needs
    # a separately identified experiment rather than mutating saved provenance.
    teacher = np.log(np.array([[.20, .79, .01]], np.float32))
    target = tf.constant(np.column_stack(([[0., 1., 0.]], teacher, [1.])), tf.float32)
    worse = tf.constant([[2., 0., 0.]])
    np.testing.assert_allclose(protected_loss()(target, worse), loss_function()(target, worse))


def test_invalid_retention_predictions_and_protection_parameters_fail_closed():
    tf = pytest.importorskip('tensorflow')
    from scripts.balloon_positive_protection import protected_loss
    with pytest.raises(ValueError, match='settings'):
        protected_loss(cap=.7)
    with pytest.raises(ValueError, match='receipt'):
        retention(np.array([[np.nan, 0, 0]]), np.zeros((1, 3)), [1], [True])
    with pytest.raises(ValueError, match='receipt'):
        retention(np.zeros((1, 3)), np.zeros((2, 3)), [1], [True])
    assert tf.executing_eagerly()
