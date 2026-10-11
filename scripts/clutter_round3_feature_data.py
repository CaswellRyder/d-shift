"""Rebuild matched TRAIN tensors; scene predictions never become hard labels."""
import hashlib
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256
from scripts.blue_object_round3_data import checked_extra, REVIEW, QUEUE, ROOT
from scripts.balloon_scene_constraints import scene_pixels, RAW
from scripts.fit_blue_prototype_rejection import checked_inputs
from scripts.grid_clutter_data import load_crop
from scripts.train_batch3_retraining import LABELS, choose_controls

PROTOCOL = Path('configs/clutter-round3-feature-20261011.json')


def tensors(arm):
    p = read_json(PROTOCOL)
    if arm not in p['arms'] or sha256(p['parent']) != p['parent_sha256']:
        raise ValueError('Unknown arm or changed parent')
    rows, labels, old, _, _, previous = checked_inputs()
    extra = checked_extra()
    if ({r['source_image'] for r in rows} & {r['source_image'] for r in extra}
            or {r['sha256'] for r in rows} & {r['sha256'] for r in extra}):
        raise ValueError('New negatives overlap existing curriculum')
    chosen = extra if arm == 'admit' else choose_controls(rows, extra)
    added = [dict(r, preserve=False, view='original')
             for _ in range(p['new_repetitions']) for r in chosen]
    scene, scene_ids = scene_pixels()
    probes = np.stack([load_crop(r)[0] for r in extra])
    probe_labels = np.array([LABELS.index(r['label']) for r in extra])
    x = np.concatenate([old, np.stack([load_crop(r)[0] for r in added]), scene])
    # -1 is explicitly unlabeled, not background. The loss never uses its hard label.
    y = np.r_[labels, [LABELS.index(r['label']) for r in added], np.full(len(scene), -1)]
    preserve = np.array([r['preserve'] for r in rows] + [False]*(len(added)+len(scene)), bool)
    scene_mask = np.arange(len(x)) >= len(old)+len(added)
    weights = np.where(scene_mask | preserve, 1., p['hard_weight']).astype(np.float32)
    if (len(old), len(extra), len(scene), len(x)) != (2074, 13, 385, p['rows']):
        raise ValueError('Matched curriculum changed')
    identities = [{k: r[k] for k in ('source_image', 'sha256', 'label', 'preserve', 'view')}
                  for r in rows+added]
    paths = [PROTOCOL, REVIEW, QUEUE/'review.json', ROOT/'acquisition.json',
        ROOT/'license-verification.json', previous, RAW, __file__,
        'scripts/blue_object_round3_data.py', 'scripts/balloon_scene_constraints.py',
        'scripts/fit_blue_prototype_rejection.py', 'scripts/grid_clutter_data.py',
        'scripts/train_batch3_retraining.py', 'scripts/balloon_all_correct_protection.py',
        'scripts/balloon_positive_protection.py', 'scripts/train_balloon_reviewed_clutter.py',
        'research/balloon-red-blue-20261008/blue-object-round3-admission-20261011.json']
    return dict(p=p, rows=rows, x=x, y=y, preserve=preserve, scene_mask=scene_mask,
        weights=weights, probes=probes, probe_labels=probe_labels,
        scene_ids=scene_ids, sample_identity=identities,
        sources_sha256={str(path): sha256(path) for path in paths})


def hashes(data):
    return {k: hashlib.sha256(data[k].tobytes()).hexdigest()
            for k in ('x', 'y', 'preserve', 'scene_mask', 'weights', 'probes', 'probe_labels')}


def targets(labels, teacher, preserve, scene):
    labels, teacher = np.asarray(labels), np.asarray(teacher)
    preserve, scene = np.asarray(preserve, bool), np.asarray(scene, bool)
    n = len(labels)
    if (teacher.shape != (n, 3) or preserve.shape != (n,) or scene.shape != (n,)
            or not np.isfinite(teacher).all() or np.any(preserve & scene)
            or np.any(labels[scene] != -1) or np.any((labels[~scene]<0) | (labels[~scene]>2))):
        raise ValueError('Invalid supervised/scene target boundary')
    hard = np.zeros((n, 3), np.float32)
    indexes = np.flatnonzero(~scene)
    hard[indexes, labels[indexes]] = 1
    return np.column_stack([hard, teacher, preserve, scene]).astype(np.float32)


def loss_function(scene_kl_weight=8., scene_logit_weight=.5):
    import tensorflow as tf
    from scripts.balloon_all_correct_protection import all_correct_loss
    supervised = all_correct_loss(8., .9, .8)

    def loss(target, logits):
        teacher = target[:, 3:6]
        kl = tf.reduce_sum(tf.nn.softmax(teacher) *
            (tf.nn.log_softmax(teacher)-tf.nn.log_softmax(logits)), axis=1)
        def centered(z):
            return z-tf.reduce_mean(z, axis=1, keepdims=True)
        mse = tf.reduce_mean(tf.square(centered(teacher)-centered(logits)), axis=1)
        return tf.where(target[:, 7] > .5, scene_kl_weight*kl+scene_logit_weight*mse,
                        supervised(target[:, :7], logits))

    return loss
