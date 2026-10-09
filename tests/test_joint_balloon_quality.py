import numpy as np
import pytest

from dtr.data import read_json
from scripts.refine_balloon_quality import check_config, refinement_loss, split_refinement
from scripts.train_balloon_quality import fold_ids
from scripts.evaluate_joint_balloon_quality import refresh_predictions


def sample(group):
    return dict(source_group=group, source_image=group+'.jpg', source_sha256=group+'hash', split='train')


def partition():
    rows = [sample(str(i)) for i in range(30)]
    return rows, [r for r, f in zip(rows, fold_ids(rows)) if f == 0]


def test_whole_quality_holdout_sources_excluded_from_all_refinement_inputs():
    rows, held = partition()
    aliases = [dict(sample('alias'), **{key: held[0][key]})
               for key in ('source_group', 'source_image', 'source_sha256')]
    classes, train, val = split_refinement(rows+aliases+[dict(sample('other'), split='test')], rows)
    assert val == held and classes == train
    assert set(r['source_group'] for r in classes).isdisjoint(r['source_group'] for r in held)


@pytest.mark.parametrize('fold', [-1, 5, True, .5])
def test_invalid_holdout_fold_rejected(fold):
    with pytest.raises(ValueError, match='fold'):
        split_refinement([], [], fold)


def test_cross_fold_source_alias_and_empty_class_pool_rejected():
    rows, held = partition()
    with pytest.raises(ValueError, match='Empty'):
        split_refinement([], rows)
    train_row = next(r for r in rows if r not in held)
    altered = [dict(r, source_sha256=held[0]['source_sha256']) if r is train_row else r for r in rows]
    with pytest.raises(ValueError, match='leakage'):
        split_refinement(rows, altered)
    with pytest.raises(ValueError, match='examples'):
        split_refinement(rows, held)


@pytest.mark.parametrize('key,value', [('epochs', 0), ('epochs', True), ('batch_size', 65),
    ('learning_rate', .1), ('learning_rate', float('nan')), ('temperature', 0),
    ('quality_weight', 2), ('preservation_weight', 0), ('deployment_approved', True),
    ('checkpoint_selection', 'best development recall')])
def test_bounded_config_rejects_unsafe_or_unmatched_settings(key, value):
    config = read_json('configs/balloon-quality-joint-20261008.json')
    config[key] = value
    with pytest.raises(ValueError):
        check_config(config, 'joint')


def test_modes_and_config_accepted():
    config = read_json('configs/balloon-quality-joint-20261008.json')
    for mode in ('joint', 'head_only'):
        check_config(config, mode)
    with pytest.raises(ValueError):
        check_config(config, 'unbounded')


def test_loss_masks_unlabeled_quality_but_preserves_all_class_logits():
    tf = pytest.importorskip('tensorflow')
    target = tf.constant([[1., 2., 3., .5, 0.], [1., 2., 3., .5, 1.]])
    output = tf.Variable([[1., 2., 3., 2.], [1., 2., 3., 2.]])
    loss = refinement_loss(4., 2.)
    with tf.GradientTape() as tape:
        values = loss(target, output)
        total = tf.reduce_sum(values)
    gradient = tape.gradient(total, output).numpy()
    np.testing.assert_allclose(values.numpy(), [0., 4.5], atol=1e-6)
    assert gradient[0, 3] == 0 and gradient[1, 3] == 6
    changed = output.numpy()
    changed[0, 0] += 2
    assert loss(target, tf.constant(changed)).numpy()[0] > 0


@pytest.mark.parametrize('joint', [False, True])
def test_frozen_classifier_allows_feature_gradients_only_in_joint_mode(joint):
    keras = pytest.importorskip('keras')
    inp = keras.Input((4,))
    features = keras.layers.Dense(3, name='features')(inp)
    parent = keras.Model(inp, keras.layers.Dense(3, name='classifier')(features))
    parent.trainable = joint
    parent.get_layer('classifier').trainable = False
    quality = keras.layers.Dense(1, name='quality')(features)
    model = keras.Model(inp, keras.layers.Concatenate()([parent.output, quality]))
    names = [v.path for v in model.trainable_weights]
    assert not any('classifier' in n for n in names)
    assert any('quality' in n for n in names)
    assert any('features' in n for n in names) == joint


class FakePredictor:
    metadata = dict(classes=['background', 'red_balloon', 'blue_balloon'])

    def predict(self, rgb):
        return np.array([.01, .01, .98]), .75


def detection():
    return dict(box=[10, 10, 30, 30], crop_box=[8, 8, 32, 32], area=400,
        label='red_balloon', score=.9, scores=[.05, .9, .05], raw_accepted=True,
        accepted=False, suppressed=True, suppression_reason='confirmed_balloon_part',
        rejection_reason='duplicate', suppressed_by=3, suppressed_by_index=3)


def test_fresh_scores_remove_old_class_and_all_suppression_state():
    old = detection()
    rows, changes = refresh_predictions(FakePredictor(), np.zeros((240, 320, 3), np.uint8), [old])
    assert rows[0]['accepted'] and rows[0]['raw_accepted'] and not rows[0]['suppressed']
    assert rows[0]['label'] == 'blue_balloon' and rows[0]['box_quality'] == .75
    for stale in ('suppression_reason', 'rejection_reason', 'suppressed_by', 'suppressed_by_index'):
        assert stale not in rows[0]
    assert changes['label_changes'] == 1 and changes['raw_acceptance_changes'] == 0
    assert old['suppressed'] and old['label'] == 'red_balloon'


def test_fresh_predictions_run_ordinary_nms_again():
    rows, _ = refresh_predictions(FakePredictor(), np.zeros((240, 320, 3), np.uint8),
                                 [detection(), detection()])
    assert sum(r['accepted'] for r in rows) == 1
    assert all(r['raw_accepted'] for r in rows)
    assert rows[1]['rejection_reason'] == 'duplicate'


@pytest.mark.parametrize('field,value', [('crop_box', [-1, 0, 10, 10]),
    ('crop_box', [0., 0, 10, 10]), ('scores', [float('nan'), 0, 0]), ('scores', [0, 1])])
def test_malformed_saved_prediction_fails(field, value):
    with pytest.raises(ValueError):
        refresh_predictions(FakePredictor(), np.zeros((240, 320, 3), np.uint8),
                            [dict(detection(), **{field: value})])
