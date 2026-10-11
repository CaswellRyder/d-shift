import numpy as np
import pytest

from scripts.scene_decision_retention import decision_loss, loss_function
from scripts.scene_teacher_supervision import loss_function as prior_loss


def logits(probabilities):
    tf = pytest.importorskip('tensorflow')
    return tf.constant(np.log(probabilities), dtype=tf.float32)


def test_loss_zero_at_teacher_and_confidence_can_relax():
    teacher = logits([[.005, .99, .005], [.99, .005, .005], [.1, .5, .4], [.198, .801, .001]])
    np.testing.assert_allclose(decision_loss(teacher, teacher), 0., atol=1e-10)
    relaxed = logits([[.07, .86, .07], [.20, .40, .40], [.15, .45, .4], [.19, .805, .005]])
    np.testing.assert_allclose(decision_loss(teacher, relaxed), 0., atol=1e-10)


def test_lost_positive_and_new_wrong_acceptance_are_penalized():
    teacher = logits([[.05, .9, .05], [.8, .1, .1], [.8, .1, .1]])
    changed = logits([[.15, .7, .15], [.05, .9, .05], [.05, .05, .9]])
    assert np.all(decision_loss(teacher, changed).numpy() > 0)


def test_shift_invariant_and_finite_gradients():
    tf = pytest.importorskip('tensorflow')
    teacher = logits([[.05, .9, .05]])
    changed = tf.Variable(logits([[.2, .6, .2]]))
    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(decision_loss(teacher, changed))
    assert np.isfinite(tape.gradient(loss, changed).numpy()).all()
    np.testing.assert_allclose(decision_loss(teacher, changed), decision_loss(teacher+10, changed-10), rtol=1e-5)


def test_only_unreviewed_scenes_change_loss():
    tf = pytest.importorskip('tensorflow')
    target = np.zeros((3, 12), np.float32)
    target[:, 3:6] = np.log([.05, .9, .05])
    target[0, 1] = 1  # Ordinary labeled red crop.
    target[1:, 7] = 1
    target[1, 10:12] = 1  # Reviewed blue correction.
    z = logits([[.2, .6, .2]]*3)
    previous = prior_loss()(tf.constant(target), z).numpy()
    current = loss_function()(tf.constant(target), z).numpy()
    np.testing.assert_array_equal(previous[:2], current[:2])
    np.testing.assert_allclose(current[2:], decision_loss(tf.constant(target[2:, 3:6]), z[2:]))


def test_development_requires_both_decision_seeds():
    from scripts.evaluate_scene_decision_retention import development_names
    names = ['control47', 'control48', 'decision47', 'decision48']
    results = {n: dict(nonregression=True) for n in names}
    assert development_names(names, results) == names
    results['decision48']['nonregression'] = False
    assert development_names(names, results) == []
    with pytest.raises(ValueError):
        development_names(names[:-1], results)
