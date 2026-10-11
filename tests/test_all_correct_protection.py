import numpy as np
import pytest


def test_new_loss_matches_previous_for_already_accepted_and_nonpositive_rows():
    tf = pytest.importorskip('tensorflow')
    from scripts.balloon_all_correct_protection import all_correct_loss
    from scripts.balloon_positive_protection import protected_loss
    target = tf.constant([[0, 1, 0, 0, 4, 0, 1], [1, 0, 0, 4, 0, 0, 1],
                          [0, 1, 0, 4, 0, 0, 1], [0, 1, 0, 0, 1, 0, 0]], tf.float32)
    logits = tf.zeros((4, 3))
    np.testing.assert_allclose(all_correct_loss()(target, logits), protected_loss()(target, logits), atol=1e-6)


def test_new_loss_protects_correct_low_confidence_positive_without_forcing_point9():
    tf = pytest.importorskip('tensorflow')
    from scripts.balloon_all_correct_protection import all_correct_loss
    from scripts.balloon_positive_protection import protected_loss
    teacher = np.log(np.array([[.2, .79, .01]], np.float32))
    target = tf.constant(np.column_stack(([[0., 1., 0.]], teacher, [1.])), tf.float32)
    worse = tf.Variable([[2., 0., 0.]])
    with tf.GradientTape() as tape:
        penalty = tf.reduce_sum(all_correct_loss()(target, worse) - protected_loss()(target, worse))
    gradient = tape.gradient(penalty, worse).numpy()[0]
    assert float(penalty) > 0 and gradient[1] < 0 and gradient[0] > 0
    # Original confidence0.79 incurs no extra penalty: do not replace it with0.9.
    np.testing.assert_allclose(all_correct_loss()(target, tf.constant(teacher)),
                               protected_loss()(target, tf.constant(teacher)), atol=1e-6)
    np.testing.assert_allclose(all_correct_loss()(target, worse + 30), all_correct_loss()(target, worse), atol=.0001)


def test_target_requires_both_colors_precision_and_recall_not_average():
    from scripts.summarize_balloon_mask_convergence import meets_target
    passing = dict(red_balloon=dict(precision=.94, recall=.94), blue_balloon=dict(precision=.94, recall=.94))
    assert meets_target(passing)
    assert not meets_target(dict(passing, blue_balloon=dict(precision=.939, recall=1.)))
    assert not meets_target(dict(passing, red_balloon=dict(precision=1., recall=.939)))
    assert not meets_target(dict(red_balloon=passing['red_balloon']))
    assert not meets_target(dict(passing, blue_balloon=dict(precision=None, recall=0.)))
