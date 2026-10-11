import copy

import numpy as np
import pytest

from scripts.scene_teacher_supervision import correction_labels, loss_function
from scripts.clutter_round3_feature_data import loss_function as original_loss


def review_fixture():
    identities = [dict(source='train/example', crop_box=[0, 0, 10, 10]) for _ in range(2)]
    queue = dict(samples=[dict(id=i, **r, original=dict(category='empty_crop')) for i, r in enumerate(identities)])
    return dict(labels={'0': 'background'}, excluded={'1': 'Ambiguous latex'}), queue, identities


def test_explicit_exclusion_remains_unlabeled():
    r, q, ids = review_fixture()
    np.testing.assert_array_equal(correction_labels(r, q, ids), [0, -1])


@pytest.mark.parametrize('mutation', ['overlap', 'missing', 'partial', 'img', 'duplicate', 'index'])
def test_invalid_review_fails_closed(mutation):
    r, q, ids = review_fixture()
    if mutation == 'overlap':
        r['labels']['1'] = 'background'
    elif mutation == 'missing':
        r['excluded'] = {}
    elif mutation == 'partial':
        q['samples'][0]['original']['category'] = 'partial_or_mixed'
    elif mutation == 'img':
        q['samples'][0]['source'] = ids[0]['source'] = 'train/IMG_example'
    elif mutation == 'duplicate':
        q['samples'].append(copy.deepcopy(q['samples'][0]))
    else:
        q['samples'][0]['id'] = -1
    with pytest.raises(ValueError):
        correction_labels(r, q, ids)


def test_reviewed_loss_ignores_wrong_teacher_and_preserves_other_rows():
    tf = pytest.importorskip('tensorflow')
    target = np.zeros((2, 12), np.float32)
    target[:, 3:6] = [0., 10., 0.]  # Teacher incorrectly insists red.
    target[:, 7] = 1
    target[0, 10:12] = 1  # Reviewed blue, correction enabled.
    logits = tf.constant([[0., 0., 1.], [0., 0., 1.]])
    observed = loss_function()(tf.constant(target), logits).numpy()
    expected = -4 * tf.nn.log_softmax(logits)[0, 2].numpy()
    np.testing.assert_allclose(observed[0], expected)
    np.testing.assert_allclose(observed[1], original_loss()(tf.constant(target[:, :8]), logits).numpy()[1])
    target[0, 3:6] = [100., -100., 50.]
    np.testing.assert_allclose(loss_function()(tf.constant(target), logits).numpy()[0], expected)
