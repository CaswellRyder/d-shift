from collections import Counter

import numpy as np
import pytest

from scripts import train_balloon_reviewed_clutter as train


def fixtures(monkeypatch, conflicting=False):
    ordinary = [dict(label=label, sha256=str(i), path=f'{i}.png', source_image=f'frame_{i}.jpg')
                for i, label in enumerate(train.CLASSES)]
    extra = [dict(label='balloon_fragment', sha256='fragment', path='f.png', source_image='frame_f.jpg')]
    negatives = [dict(label='background', sha256=str(1 if conflicting else 0) if i == 0 else f'n{i}',
                      path=f'n{i}.png', source_image=f'frame_n{i}.jpg') for i in range(9)]
    monkeypatch.setattr(train, 'checked_data', lambda: (ordinary, extra))
    monkeypatch.setattr(train, 'checked_rows', lambda *args: negatives)


def test_matched_arms_have_identical_base_order_and_class_counts(monkeypatch):
    fixtures(monkeypatch)
    control, _ = train.data('control')
    admit, _ = train.data('admit')
    assert control[:-36] == admit[:-36]
    assert Counter(r['label'] for r in control) == Counter(r['label'] for r in admit)
    assert all(not r['preserve'] and r['label'] == 'background' for r in admit[-36:])
    assert all(r['preserve'] for r in admit[:-36])
    assert control == train.data('control')[0]


def test_conflicting_negative_label_is_rejected(monkeypatch):
    fixtures(monkeypatch, conflicting=True)
    with pytest.raises(ValueError, match='conflicts'):
        train.data('admit')


def test_invalid_mode_rejected_before_loading_data():
    with pytest.raises(ValueError, match='matched'):
        train.data('unreviewed')


def test_teacher_does_not_override_explicit_clutter_labels():
    import tensorflow as tf
    loss = train.loss_function()
    target = np.array([[1,0,0,1,2,3,0], [1,0,0,1,2,3,1]], np.float32)
    logits = tf.constant([[1.,2.,3.], [1.,2.,3.]])
    before = loss(tf.constant(target), logits).numpy()
    target[:,3:6] = [9.,-9.,-9.]
    after = loss(tf.constant(target), logits).numpy()
    assert before[0] == after[0] and before[1] != after[1]
