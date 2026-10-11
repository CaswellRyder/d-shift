"""Small TRAIN-only linear head update with explicit sufficient margin constraints."""
import numpy as np
from scipy.optimize import linprog
from scipy.special import logsumexp


def constraints(design, logits, labels, protected, new_rows, threshold=.8, new_threshold=.81):
    design, logits = np.asarray(design, float), np.asarray(logits, float)
    labels, protected, new_rows = np.asarray(labels), np.asarray(protected, bool), np.asarray(new_rows, bool)
    n, d = design.shape
    if (logits.shape != (n, 3) or labels.shape != (n,) or protected.shape != (n,) or new_rows.shape != (n,)
            or not np.isfinite(design).all() or not np.isfinite(logits).all()
            or not np.isin(labels, [0, 1, 2]).all() or not .5 < threshold < new_threshold < 1):
        raise ValueError('Invalid constraint inputs')
    a, b, identities = [], [], []
    for i in np.flatnonzero(protected | new_rows):
        truth = int(labels[i])
        others = [j for j in range(3) if j != truth]
        margins = logits[i, truth]-logits[i, others]
        if new_rows[i]:
            desired = np.full(2, np.log(2*new_threshold/(1-new_threshold))) if truth else np.full(2, .001)
        elif truth:
            # Allocate the threshold's rival mass in the parent's proportions.
            # Each rival bound jointly guarantees the required softmax mass.
            desired = margins + logsumexp(-margins)-np.log((1-threshold)/threshold)+.001
        else:
            desired = np.full(2, .001)
        for rival, current, bound in zip(others, margins, desired, strict=True):
            row = np.zeros((d, 2))
            if truth:
                row[:, truth-1] += design[i]
            if rival:
                row[:, rival-1] -= design[i]
            a.append(row.ravel())
            b.append(bound-current)
            identities.append((int(i), truth, rival))
    return np.asarray(a).reshape(-1, d*2), np.asarray(b), identities


def solve(a, b, time_limit=60.):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.ndim != 2 or b.shape != (len(a),) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Invalid linear system')
    d = a.shape[1]
    # Minimize L1 change in standardized feature coordinates; background stays fixed.
    upper = np.block([[-a, np.zeros_like(a)], [np.eye(d), -np.eye(d)], [-np.eye(d), -np.eye(d)]])
    bound = np.concatenate([-b, np.zeros(2*d)])
    result = linprog(np.r_[np.zeros(d), np.ones(d)], A_ub=upper, b_ub=bound,
        bounds=[(None, None)]*d+[(0, None)]*d, method='highs',
        options={'time_limit': time_limit, 'primal_feasibility_tolerance': 1e-8, 'dual_feasibility_tolerance': 1e-8})
    delta = result.x[:d] if result.success else None
    residual = float(np.max(b-a@delta, initial=0)) if delta is not None else None
    if delta is not None and (not np.isfinite(delta).all() or residual > 1e-6):
        raise ValueError('Solver returned an invalid constraint solution')
    return delta, dict(success=bool(result.success), status=int(result.status), message=result.message,
        iterations=int(result.nit), maximum_violation=residual,
        objective=float(result.fun) if result.success else None)
