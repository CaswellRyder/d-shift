import numpy as np
import pytest

from scripts.train_balloon_relative_size import relative_size_penalty


def inputs():
    target = np.zeros((1, 15, 20, 9), np.float32)
    prediction = np.zeros((1, 15, 20, 6), np.float32)
    target[0, 4, 5, 8] = 1
    target[0, 4, 5, 6:8] = [.1, .2]
    prediction[0, 4, 5, 4:6] = [.1, .2]
    return target, prediction


def test_size_penalty_ignores_unknown_cells_and_is_zero_for_exact_sizes():
    tf = pytest.importorskip('tensorflow')
    target, prediction = inputs()
    prediction[0, 0, 0, 4:6] = [.9, .9]
    assert float(relative_size_penalty(tf.constant(target), tf.constant(prediction))) == 0


def test_size_penalty_multiplicative_symmetry_and_scale_invariance():
    tf = pytest.importorskip('tensorflow')
    target, prediction = inputs()
    prediction[0, 4, 5, 4:6] *= 2
    a = float(relative_size_penalty(tf.constant(target), tf.constant(prediction)))
    prediction[0, 4, 5, 4:6] /= 4
    b = float(relative_size_penalty(tf.constant(target), tf.constant(prediction)))
    target[..., 6:8] /= 2
    prediction[..., 4:6] /= 2
    c = float(relative_size_penalty(tf.constant(target), tf.constant(prediction)))
    assert a == pytest.approx(.5 * np.log(2) ** 2)
    assert b == pytest.approx(a)
    assert c == pytest.approx(a)


def test_size_gradient_only_at_valid_sizes_and_correct_direction():
    tf = pytest.importorskip('tensorflow')
    target, prediction = inputs()
    prediction[0, 4, 5, 4:6] = [.2, .1]
    prediction = tf.Variable(prediction)
    with tf.GradientTape() as tape:
        loss = relative_size_penalty(tf.constant(target), prediction)
    grad = tape.gradient(loss, prediction).numpy()
    assert np.isfinite(grad).all()
    assert grad[0, 4, 5, 4] > 0
    assert grad[0, 4, 5, 5] < 0
    assert np.count_nonzero(grad) == 2


def test_no_supervised_centers_has_zero_loss_and_finite_zero_gradients():
    tf = pytest.importorskip('tensorflow')
    target, prediction = inputs()
    target[..., 8] = 0
    prediction = tf.Variable(prediction)
    with tf.GradientTape() as tape:
        loss = relative_size_penalty(tf.constant(target), prediction)
    assert float(loss) == 0
    np.testing.assert_array_equal(tape.gradient(loss, prediction).numpy(), 0)


def test_training_gate_needs_blue_gain_without_other_fit_regressions():
    from copy import deepcopy
    from scripts.summarize_balloon_relative_size import passes_training_gate
    control = {s: {c: dict(targets=10, learned=5) for c in ('red_balloon', 'blue_balloon')}
               for s in ('common_fit', 'extra_fit')}
    candidate = deepcopy(control)
    assert not passes_training_gate(control, candidate)
    candidate['extra_fit']['blue_balloon']['learned'] = 6
    assert passes_training_gate(control, candidate)
    candidate['common_fit']['red_balloon']['learned'] = 4
    assert not passes_training_gate(control, candidate)
