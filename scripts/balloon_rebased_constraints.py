"""Retain reference decisions while correcting a different feature checkpoint."""
import numpy as np

from scripts.balloon_head_constraints import constraints
from scripts.balloon_false_acceptance_constraints import constraints as rejections
from scripts.balloon_scene_constraints import retention_constraints
from scripts.train_batch3_retraining import correct


def rebase(bounds, identities, reference, current):
    """Convert reference-relative gap bounds into current-relative delta bounds.

    Positive target IDs encode (target-rival), negative IDs (rival-target).
    Background has ID0 and stays fixed in the delta parameterization, but its
    base logit may differ between the original and newly trained features.
    """
    reference, current, bounds = np.asarray(reference, float), np.asarray(current, float), np.asarray(bounds, float)
    if (reference.ndim != 2 or reference.shape[1] != 3 or current.shape != reference.shape
            or bounds.shape != (len(identities),) or not all(np.isfinite(v).all() for v in (reference, current, bounds))):
        raise ValueError('Invalid logit rebasing inputs')
    result = bounds.copy()
    for j, (index, target, rival) in enumerate(identities):
        if not (0 <= index < len(reference) and target in (-2, -1, 0, 1, 2)
                and rival in (0, 1, 2) and abs(target) != rival):
            raise ValueError('Invalid constraint identity')
        sign = -1 if target < 0 else 1
        t = abs(target)
        result[j] += sign*(reference[index, t]-reference[index, rival]-current[index, t]+current[index, rival])
    return result


def system(design, reference, current, labels):
    if design.shape != (2472, 129) or reference.shape != (2472, 3) or current.shape != reference.shape or labels.shape != (2074,):
        raise ValueError('Expected2074old+13matched+385TRAIN scene rows')
    y = np.r_[labels, np.zeros(13, int)]
    a, b, ids = constraints(design[:2087], reference[:2087], y,
        np.r_[correct(reference[:2074], labels), np.zeros(13, bool)],
        np.r_[np.zeros(2074, bool), np.ones(13, bool)])
    sa, sb, si = retention_constraints(design[2087:], reference[2087:])
    ra, rb, ri = rejections(design[:2074], reference[:2074], labels)
    ids += [(i+2087, t, r) for i, t, r in si] + ri
    a, b = np.concatenate([a, sa, ra]), np.r_[b, sb, rb]
    return a, rebase(b, ids, reference, current), np.asarray(ids)
