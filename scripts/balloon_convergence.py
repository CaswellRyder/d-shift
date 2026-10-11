"""Training-only fit trajectories, never checkpoint selection or qualification."""
import numpy as np


def snapshot(epoch, logits, labels, preserve, hard_logits, hard_labels):
    logits, hard_logits = np.asarray(logits), np.asarray(hard_logits)
    labels, hard_labels, preserve = np.asarray(labels), np.asarray(hard_labels), np.asarray(preserve)
    if (logits.shape != (len(labels),3) or hard_logits.shape != (len(hard_labels),3)
            or preserve.shape != labels.shape or preserve.dtype != np.bool_
            or not preserve.any() or not len(hard_labels)
            or not np.isfinite(logits).all() or not np.isfinite(hard_logits).all()
            or not np.isin(labels,[0,1,2]).all() or not np.isin(hard_labels,[0,1,2]).all()):
        raise ValueError('Invalid training-only probe arrays')
    exp = np.exp(hard_logits-hard_logits.max(axis=1,keepdims=True))
    probabilities = exp/exp.sum(axis=1,keepdims=True)
    return dict(epoch=epoch, ordinary_rows=int(preserve.sum()),
        ordinary_correct=int(((logits.argmax(1)==labels)&preserve).sum()),
        hard_correct=int((hard_logits.argmax(1)==hard_labels).sum()),
        hard_rows=len(hard_labels), hard_probabilities=probabilities.tolist(),
        reviewed_class_probabilities=probabilities[np.arange(len(hard_labels)),hard_labels].tolist())
