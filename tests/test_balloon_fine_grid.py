import numpy as np
import pytest

from scripts.balloon_fine_grid import encode_grid, decode, network, loss
from scripts.prepare_balloon_dense_proposals import encode
from scripts.balloon_dense_complete import encode_complete
from scripts.audit_balloon_dense_proposals import decode as coarse_decode


TRUTH = [dict(label='blue_balloon', box=[43., 65., 97., 128.]),
         dict(label='red_balloon', box=[134., 51., 197., 133.])]


@pytest.mark.parametrize('complete', (False, True))
def test_coarse_grid_encoder_exactly_reproduces_control(complete):
    old = encode_complete if complete else encode
    expected, collisions = old(TRUTH, [[240, 180, 320, 240]])
    actual, count = encode_grid(TRUTH, [[240, 180, 320, 240]], grid=(15, 20), complete=complete)
    np.testing.assert_array_equal(actual, expected)
    assert collisions == count


def test_finer_grid_restores_colliding_geometry_supervision():
    truth = [dict(label='blue_balloon', box=[33, 33, 39, 39]),
             dict(label='red_balloon', box=[41, 41, 47, 47])]
    coarse, count = encode_grid(truth, [], grid=(15, 20))
    fine, new_count = encode_grid(truth, [])
    assert count == 1 and new_count == 0
    assert coarse[..., 8].sum() == 0 and fine[..., 8].sum() == 2
    assert fine[0, 0, 2:4].sum() == 0  # Unknown remains ignored.


def test_conflicting_fine_background_is_excluded_not_positive():
    from scripts.balloon_fine_grid import encode_fine_safe
    target, skipped = encode_fine_safe(TRUTH, [[40, 64, 100, 136], [240, 180, 320, 240]])
    expected, _ = encode_grid(TRUTH, [[240, 180, 320, 240]])
    np.testing.assert_array_equal(target, expected)
    assert skipped == [[40, 64, 100, 136]]
    with pytest.raises(ValueError, match='Invalid explicit background'):
        encode_fine_safe(TRUTH, [[-1, 0, 100, 100]])


def test_coarse_decoder_exactly_reproduces_control():
    rng = np.random.default_rng(47)
    values = rng.uniform(0, 1, (15, 20, 6)).astype(np.float32)
    values[..., :2] = rng.uniform(-8, 4, (15, 20, 2))
    assert decode(values) == coarse_decode(values)


def test_fine_target_geometry_roundtrips():
    encoded, _ = encode_grid(TRUTH, [])
    values = np.zeros((30, 40, 6), np.float32)
    values[..., :2] = -80
    for y, x in np.argwhere(encoded[..., 8] > 0):
        group = np.argmax(encoded[y, x, :2])
        values[y, x, group] = 8
        values[y, x, 2:] = encoded[y, x, 4:8]
    found = decode(values)
    assert len(found) == 2
    for row in TRUTH:
        group = int(row['label'] == 'blue_balloon')
        got = next(r for r in found if r['color_group'] == group)
        np.testing.assert_allclose(got['box'], row['box'], atol=1e-4)


def test_parameter_count_and_initial_weights_match_control():
    pytest.importorskip('tensorflow')
    import keras
    from scripts.train_balloon_dense_proposals import network as old_network
    keras.utils.set_random_seed(47)
    old = old_network()
    keras.utils.set_random_seed(47)
    new = network()
    assert new.count_params() == old.count_params() == 2342
    assert new.output_shape == (None, 30, 40, 6)
    for a, b in zip(old.get_weights(), new.get_weights(), strict=True):
        np.testing.assert_array_equal(a, b)


def test_generic_loss_matches_coarse_and_masks_unknown_geometry():
    tf = pytest.importorskip('tensorflow')
    from scripts.balloon_dense_iou_loss import giou_loss
    target, _ = encode_grid(TRUTH, [], grid=(15, 20))
    pred = np.random.default_rng(47).uniform(0, 1, (1, 15, 20, 6)).astype(np.float32)
    target = tf.constant(target[None])
    assert float(loss(target, tf.constant(pred))) == pytest.approx(float(giou_loss(target, tf.constant(pred))))
    fine, _ = encode_grid(TRUTH, [])
    pred = tf.Variable(np.full((1, 30, 40, 6), .5, np.float32))
    with tf.GradientTape() as tape:
        value = loss(tf.constant(fine[None]), pred)
    gradient = tape.gradient(value, pred).numpy()
    assert np.isfinite(gradient).all()
    np.testing.assert_array_equal(gradient[0, :, :, 2:][fine[..., 8] == 0], 0)
