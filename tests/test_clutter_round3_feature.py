import numpy as np
import pytest

from scripts.clutter_round3_feature_data import targets, loss_function


def test_scene_targets_are_not_background_annotations():
    result = targets(np.array([0, 2, -1]), np.zeros((3, 3)), [True, True, False], [False, False, True])
    np.testing.assert_array_equal(result[:, :3], [[1, 0, 0], [0, 0, 1], [0, 0, 0]])
    np.testing.assert_array_equal(result[:, 7], [0, 0, 1])
    with pytest.raises(ValueError):
        targets(np.array([-1]), np.zeros((1, 3)), [False], [False])
    with pytest.raises(ValueError):
        targets(np.array([0]), np.zeros((1, 3)), [False], [True])
    with pytest.raises(ValueError):
        targets(np.array([-1]), np.zeros((1, 3)), [True], [True])


def test_scene_loss_is_zero_at_teacher_and_shift_invariant():
    tf = pytest.importorskip('tensorflow')
    teacher = np.array([[1., 3., 0.]], np.float32)
    target = tf.constant(targets(np.array([-1]), teacher, [False], [True]))
    loss = loss_function()
    assert float(loss(target, tf.constant(teacher))[0]) == pytest.approx(0, abs=1e-6)
    assert float(loss(target, tf.constant(teacher+20))[0]) == pytest.approx(0, abs=1e-6)
    assert float(loss(target, tf.constant([[3., 0., 1.]]))[0]) > 0


def test_supervised_loss_still_learns_background():
    tf = pytest.importorskip('tensorflow')
    teacher = np.array([[0., 4., 0.]], np.float32)
    target = tf.constant(targets(np.array([0]), teacher, [False], [False]))
    loss = loss_function()
    assert float(loss(target, tf.constant([[4., 0., 0.]]))[0]) < float(loss(target, tf.constant(teacher))[0])


def test_scene_decision_loss_blocks_otherwise_passing_training_gate():
    from scripts.audit_clutter_round3_feature import recompute
    before = np.tile([4., 0., 0.], (2511, 1))
    after = before.copy()
    labels = np.r_[np.zeros(2126, dtype=int), np.full(385, -1)]
    scene = np.arange(2511) >= 2126
    probes = np.tile([4., 0., 0.], (13, 1))
    probe_labels = np.zeros(13, dtype=int)
    assert recompute(before, after, labels, probes, probe_labels, scene)['training_gate_pass']
    after[-1] = [0., 0., 4.]
    result = recompute(before, after, labels, probes, probe_labels, scene)
    assert not result['training_gate_pass']
    assert result['scene_changed_rows'] == [384]


def test_both_seeds_and_matched_controls_required():
    from scripts.audit_clutter_round3_feature import eligible
    results = {f'{a}{s}': dict(training_gate_pass=True) for a in ('control', 'admit') for s in (47, 48)}
    assert eligible(results, [47, 48])
    results['admit48']['training_gate_pass'] = False
    assert not eligible(results, [47, 48])
    del results['control47']
    with pytest.raises(ValueError):
        eligible(results, [47, 48])


def test_replay_reselects_instead_of_retaining_stale_suppression():
    from scripts.diagnose_clutter_round3_feature import replay, check_replayed_parent
    frame = dict(id=0, source='train/a', frame_sha256='abc', truth=[], detections=[
        dict(box=[10, 10, 20, 20], crop_box=[8, 8, 22, 22], accepted=False,
             suppressed=True, raw_accepted=True, rejection_reason='old')])
    result = replay([frame], np.array([[.01, .98, .01]]))
    found = result['frames'][0]['detections'][0]
    assert found['accepted'] and found['raw_accepted'] and not found['suppressed']
    assert 'rejection_reason' not in found
    check_replayed_parent(result['frames'][0], result['frames'][0])
    bad = dict(result['frames'][0], counts={})
    with pytest.raises(ValueError):
        check_replayed_parent(result['frames'][0], bad)
    with pytest.raises(ValueError):
        replay([frame], np.empty((0, 3)))
