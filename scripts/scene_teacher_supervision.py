"""Correct explicitly reviewed TRAIN teacher mistakes, never implicit pseudo-labels."""
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256
from scripts.clutter_round3_feature_data import tensors as original_tensors, targets as original_targets, loss_function as original_loss
from scripts.train_batch3_retraining import LABELS

REVIEW = Path('configs/scene-teacher-review-20261011.json')
PROTOCOL = Path('configs/scene-teacher-supervision-20261011.json')


def correction_labels(review, queue, identities):
    labels = np.full(len(identities), -1, dtype=np.int64)
    samples = {str(r['id']): r for r in queue['samples']}
    if (len(samples) != len(queue['samples'])
            or any(not key.isdecimal() or not 0 <= int(key) < len(identities) for key in samples)
            or set(review['labels']) & set(review['excluded'])
            or set(review['labels']) | set(review['excluded']) != set(samples)):
        raise ValueError('Review must account for every disagreement exactly once')
    for key, value in review['labels'].items():
        row, identity = samples[key], identities[int(key)]
        if (value not in LABELS or any(row[k] != identity[k] for k in identity)
                or not row['source'].startswith('train/') or 'IMG_' in row['source']):
            raise ValueError('Changed review identity or TRAIN boundary')
        category = row['original']['category']
        if category != ('empty_crop' if value == 'background' else 'matched_' + value):
            raise ValueError('Review conflicts with complete-scene geometry')
        labels[int(key)] = LABELS.index(value)
    return labels


def tensors():
    data = original_tensors('admit')
    p, review = read_json(PROTOCOL), read_json(REVIEW)
    if sha256(review['queue']) != review['queue_sha256']:
        raise ValueError('Review queue changed')
    queue = read_json(review['queue'])
    corrected = correction_labels(review, queue, data['scene_ids'])
    if np.count_nonzero(corrected >= 0) != 10:
        raise ValueError('Require exactly10 reviewed corrections')
    for row in queue['samples']:
        if sha256(row['source_path']) != row['frame_sha256'] or sha256(Path(review['queue']).parent / row['path']) != row['sha256']:
            raise ValueError('Reviewed pixels changed')
    if any(p[k] != data['p'][k] for k in ('epochs', 'batch_size', 'optimizer_steps', 'learning_rate', 'cosine_alpha')):
        raise ValueError('Matched schedule changed')
    data['correction_labels'] = np.r_[np.full(2126, -1), corrected]
    for path in (REVIEW, PROTOCOL, __file__, review['queue'],
                 'scripts/prepare_scene_teacher_review.py', 'scripts/balloon_scene_label_audit.py'):
        data['sources_sha256'][str(path)] = sha256(path)
    return data


def targets(data, teacher):
    base = original_targets(data['y'], teacher, data['preserve'], data['scene_mask'])
    y = data['correction_labels']
    hard = np.zeros((len(y), 3), np.float32)
    mask = y >= 0
    hard[np.flatnonzero(mask), y[mask]] = 1
    return np.column_stack([base, hard, mask]).astype(np.float32)


def loss_function():
    import tensorflow as tf
    original = original_loss()
    weight = read_json(PROTOCOL)['review_weight']

    def loss(target, logits):
        corrected = -tf.reduce_sum(target[:, 8:11] * tf.nn.log_softmax(logits), axis=1)
        return tf.where(target[:, 11] > .5, weight * corrected, original(target[:, :8], logits))

    return loss
