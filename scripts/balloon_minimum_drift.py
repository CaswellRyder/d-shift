"""Convex TRAIN-logit drift minimization subject to frozen retention bounds."""
import time

import numpy as np
from scipy.optimize import LinearConstraint, minimize, nnls


def quadratic(design, ridge=.001):
    design = np.asarray(design, float)
    if design.ndim != 2 or not len(design) or not np.isfinite(design).all() or ridge <= 0:
        raise ValueError('Invalid TRAIN design or ridge')
    # Feature-major, red/blue-interleaved parameters; background stays fixed.
    return np.kron(design.T@design/len(design)+ridge*np.eye(design.shape[1]), np.eye(2))


def certificate(h, a, b, delta):
    gradient, slack = h@delta, a@delta-b
    active = np.flatnonzero(slack <= 1e-6)
    dual = np.zeros(len(b))
    if len(active):
        dual[active], _ = nnls(a[active].T, gradient, maxiter=max(1000, 10*len(active)))
    return dict(maximum_violation=float(np.max(-slack, initial=0)),
        stationarity=float(np.max(np.abs(gradient-a.T@dual), initial=0)),
        complementarity=float(np.max(np.abs(slack*dual), initial=0)),
        objective=float(.5*delta@h@delta), active_constraints=len(active))


def solve(a, b, h, initial, time_limit=120.):
    a, b, h, initial = [np.asarray(v, float) for v in (a, b, h, initial)]
    if (a.ndim != 2 or b.shape != (len(a),) or initial.shape != (a.shape[1],)
            or h.shape != (len(initial), len(initial))
            or not all(np.isfinite(v).all() for v in (a, b, h, initial))
            or not np.allclose(h, h.T, atol=1e-12, rtol=0)
            or np.linalg.eigvalsh(h).min() <= 0 or np.max(b-a@initial, initial=0) > 1e-6):
        raise ValueError('Require a positive definite objective and feasible initial solution')
    start = time.monotonic()

    def callback(x):
        if time.monotonic()-start > time_limit:
            raise TimeoutError('Bounded minimum-drift fit exceeded time limit')

    result = minimize(lambda x: .5*x@h@x, initial, jac=lambda x: h@x,
        constraints=[LinearConstraint(a, b, np.inf)], method='SLSQP', callback=callback,
        options={'maxiter': 500, 'ftol': 1e-10})
    proof = certificate(h, a, b, result.x)
    passed = bool(result.success and proof['maximum_violation'] <= 1e-6
                  and proof['stationarity'] <= 1e-4 and proof['complementarity'] <= 1e-4)
    report = dict(success=passed, status=int(result.status), message=result.message,
        iterations=int(result.nit), elapsed_s=time.monotonic()-start,
        initial_objective=float(.5*initial@h@initial), **proof)
    return result.x if passed else None, report
