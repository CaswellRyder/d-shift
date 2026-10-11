"""Sufficient TRAIN bounds against newly accepted wrong-color predictions."""
import numpy as np

from dtr.runtime import softmax


def constraints(design, logits, labels, threshold=.8):
    design, logits, labels = np.asarray(design, float), np.asarray(logits, float), np.asarray(labels)
    if (design.ndim != 2 or logits.shape != (len(design), 3) or labels.shape != (len(design),)
            or not np.isin(labels, [0, 1, 2]).all() or not .5 < threshold < 1
            or not np.isfinite(design).all() or not np.isfinite(logits).all()):
        raise ValueError('Invalid labeled TRAIN inputs')
    p = softmax(logits)
    a, b, ids = [], [], []
    bound = np.log((threshold-.001)/(1-threshold+.001))
    for i, truth in enumerate(labels):
        for target in (1, 2):
            if target == truth or (p[i].argmax() == target and p[i, target] >= threshold):
                continue  # Existing false accepts may be corrected, but are not protected.
            rival = max((j for j in range(3) if j != target), key=lambda j: logits[i, j])
            row = np.zeros((design.shape[1], 2))
            row[:, target-1] -= design[i]
            if rival:
                row[:, rival-1] += design[i]
            a.append(row.ravel())
            b.append(logits[i, target]-logits[i, rival]-bound)
            ids.append((i, -target, rival))
    return np.asarray(a).reshape(-1, design.shape[1]*2), np.asarray(b), ids
