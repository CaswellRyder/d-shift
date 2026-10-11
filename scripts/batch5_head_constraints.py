"""Compose explicit labeled-crop and unlabeled-proposal TRAIN retention bounds."""
import numpy as np

from scripts.balloon_head_constraints import constraints
from scripts.balloon_false_acceptance_constraints import constraints as rejections
from scripts.balloon_scene_constraints import retention_constraints
from scripts.train_batch3_retraining import correct


def assemble(design, logits, labels, original_count, chosen_count, old_scene_count, exempt):
    design, logits, labels = np.asarray(design, float), np.asarray(logits, float), np.asarray(labels)
    chosen_start = len(labels)
    scene_start = chosen_start+chosen_count
    batch_start = scene_start+old_scene_count
    if (design.ndim != 2 or logits.shape != (len(design), 3)
            or not 0 < original_count <= chosen_start or chosen_count <= 0 or old_scene_count <= 0
            or not batch_start <= exempt < len(design) or not np.isin(labels, [0, 1, 2]).all()
            or not np.isfinite(design).all() or not np.isfinite(logits).all()):
        raise ValueError('Invalid TRAIN head-repair groups or exempt proposal')
    matrices, bounds, identities = [], [], []

    def add(a, b, ids, mapping, group):
        matrices.append(a)
        bounds.append(b)
        identities.extend((group, int(mapping[i]), t, r) for i, t, r in ids)

    y = np.r_[labels, np.full(chosen_count, 2)]
    protected = np.r_[correct(logits[:chosen_start], labels), np.zeros(chosen_count, bool)]
    new_rows = np.arange(scene_start) >= chosen_start
    add(*constraints(design[:scene_start], logits[:scene_start], y, protected, new_rows),
        np.arange(scene_start), 0)
    add(*rejections(design[:chosen_start], logits[:chosen_start], labels), np.arange(chosen_start), 1)
    scene_indices = np.array(list(range(scene_start, batch_start))+
                             [i for i in range(batch_start, len(design)) if i != exempt], dtype=int)
    add(*retention_constraints(design[scene_indices], logits[scene_indices]), scene_indices, 2)
    objective_indices = np.r_[np.arange(chosen_start), scene_indices]
    return (np.concatenate(matrices), np.concatenate(bounds), np.asarray(identities, int),
            objective_indices)
