"""Retain original TRAIN decisions using a separately updated frozen representation."""
import numpy as np

from scripts.balloon_head_constraints import constraints
from scripts.balloon_false_acceptance_constraints import constraints as rejection_constraints
from scripts.balloon_scene_constraints import retention_constraints
from scripts.train_batch3_retraining import correct


def rebase(bounds, identities, reference, current):
    """Translate sufficient logit-gap bounds to a different starting head output."""
    reference, current = np.asarray(reference, float), np.asarray(current, float)
    result = np.asarray(bounds, float).copy()
    if (reference.shape != current.shape or reference.ndim != 2 or reference.shape[1] != 3
            or not np.isfinite(reference).all() or not np.isfinite(current).all()
            or result.shape != (len(identities),) or not np.isfinite(result).all()):
        raise ValueError('Invalid reference/current logit bounds')
    difference = reference-current
    for j, (i, target, rival) in enumerate(identities):
        if not 0 <= i < len(reference) or target not in (-2, -1, 0, 1, 2) or rival not in (0, 1, 2):
            raise ValueError('Invalid bound identity')
        positive, negative = (target, rival) if target >= 0 else (rival, -target)
        result[j] += difference[i, positive]-difference[i, negative]
    return result


def assemble(design, reference, current, labels, chosen_labels, old_scene_count, exempt):
    design, reference, current = np.asarray(design, float), np.asarray(reference, float), np.asarray(current, float)
    labels, chosen_labels = np.asarray(labels), np.asarray(chosen_labels)
    start = len(labels)
    scene_start = start+len(chosen_labels)
    batch_start = scene_start+old_scene_count
    if (design.ndim != 2 or reference.shape != (len(design), 3) or current.shape != reference.shape
            or not np.isfinite(design).all() or not np.isfinite(reference).all() or not np.isfinite(current).all()
            or not start or not len(chosen_labels) or old_scene_count <= 0
            or len(exempt) != 2 or len(set(exempt)) != 2
            or any(not batch_start <= i < len(design) for i in exempt)
            or not np.isin(labels, [0, 1, 2]).all() or not np.isin(chosen_labels, [1, 2]).all()):
        raise ValueError('Invalid TRAIN groups or reviewed-target exemptions')
    matrices, bounds, identities = [], [], []

    def add(builder, indexes, group, *args):
        a, b, ids = builder(design[indexes], reference[indexes], *args)
        matrices.append(a)
        bounds.append(rebase(b, ids, reference[indexes], current[indexes]))
        identities.extend((group, int(indexes[i]), t, r) for i, t, r in ids)

    y = np.r_[labels, chosen_labels]
    protected = np.r_[correct(reference[:start], labels), np.zeros(len(chosen_labels), bool)]
    add(constraints, np.arange(scene_start), 0, y, protected, np.arange(scene_start) >= start)
    add(rejection_constraints, np.arange(start), 1, labels)
    scenes = np.array([i for i in range(scene_start, len(design)) if i not in exempt], int)
    add(retention_constraints, scenes, 2)
    objective = np.r_[np.arange(start), scenes]
    return np.concatenate(matrices), np.concatenate(bounds), np.asarray(identities, int), objective
