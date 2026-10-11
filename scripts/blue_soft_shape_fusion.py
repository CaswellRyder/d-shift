"""TRAIN-only regularized shape correction to frozen blue/background odds."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def odds(probabilities):
    p = np.asarray(probabilities, np.float64)
    if (p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all()
            or np.any(p < 0) or not np.allclose(p.sum(axis=1), 1., atol=1e-6, rtol=0)):
        raise ValueError('Require normalized finite probabilities')
    return np.log(np.maximum(p[:, 2], 1e-12))-np.log(np.maximum(p[:, 0], 1e-12))


def fit(x, labels, probabilities):
    x, labels = np.asarray(x, np.float64), np.asarray(labels)
    base = odds(probabilities)
    if (x.ndim != 2 or labels.shape != (len(x),) or base.shape != (len(x),)
            or not np.isfinite(x).all() or not np.isin(labels, [0, 1, 2]).all()):
        raise ValueError('Invalid shape training inputs')
    keep = labels != 1
    x, labels, base = x[keep], labels[keep], base[keep]
    y = (labels == 2).astype(np.float64)
    if not np.any(y == 0) or not np.any(y == 1):
        raise ValueError('Require blue and background fitting examples')
    weights = np.where(y > 0, .5/y.sum(), .5/(len(y)-y.sum()))
    mean = np.sum(weights[:, None]*x, axis=0)
    scale = np.maximum(np.sqrt(np.sum(weights[:, None]*(x-mean)**2, axis=0)), .001)
    z = np.column_stack(((x-mean)/scale, np.ones(len(x))))

    def objective(coefficients):
        logit = base+z@coefficients
        loss = np.sum(weights*(np.logaddexp(0, logit)-y*logit))+.05*np.sum(coefficients**2)
        gradient = z.T@(weights*(expit(logit)-y))+.1*coefficients
        return loss, gradient

    result = minimize(objective, np.zeros(z.shape[1]), jac=True, method='L-BFGS-B',
                      options=dict(maxiter=500, ftol=1e-12, gtol=1e-8))
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError(f'Shape fusion fit failed: {result.message}')
    return dict(mean=mean, scale=scale, coefficients=result.x[:-1], bias=np.array(result.x[-1]),
                iterations=np.array(result.nit), objective=np.array(result.fun))


def adjust(probabilities, features, model):
    p, x = np.asarray(probabilities, np.float64), np.asarray(features, np.float64)
    base = odds(p)
    if (x.shape != (len(p), len(model['mean'])) or not np.isfinite(x).all()
            or not all(np.isfinite(v).all() for v in model.values()) or np.any(model['scale'] <= 0)):
        raise ValueError('Invalid shape model/features')
    delta = (x-model['mean'])/model['scale']@model['coefficients']+model['bias']
    result = p.copy()
    take = (p.argmax(axis=1) != 1) & (delta != 0)
    mass = p[take, 0]+p[take, 2]
    result[take, 2] = mass*expit(base[take]+delta[take])
    result[take, 0] = mass-result[take, 2]
    return result
