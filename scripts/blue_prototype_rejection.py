"""Compact-support negative prototypes in frozen features; TRAIN-only fitting."""
import numpy as np


def fit(features, labels, centers, logits):
    f, c, z = (np.asarray(v, np.float64) for v in (features, centers, logits))
    labels = np.asarray(labels)
    if (f.ndim != 2 or c.ndim != 2 or f.shape[1] != c.shape[1] or z.shape != (len(c),3)
            or labels.shape != (len(f),) or not np.isin(labels,[0,1,2]).all()
            or not np.any(labels > 0) or not len(c) or not all(np.isfinite(a).all() for a in (f,c,z))):
        raise ValueError('Require finite features, positive labels and background-center logits')
    mean, scale = f.mean(axis=0), np.maximum(f.std(axis=0), .001)
    x, normalized = (f-mean)/scale, (c-mean)/scale
    distances = np.sum((normalized[:,None,:]-x[labels > 0][None,:,:])**2, axis=2)
    radii2 = .25*distances.min(axis=1)
    return dict(mean=mean, scale=scale, centers=normalized, radii2=radii2,
        amplitude=np.maximum(0., z[:,2]-z[:,0]+8.))


def correction(features, bank):
    x = np.asarray(features, np.float64)
    if (x.ndim != 2 or x.shape[1:] != bank['mean'].shape or not np.isfinite(x).all()
            or not all(np.isfinite(v).all() for v in bank.values())
            or np.any(bank['scale'] <= 0) or np.any(bank['radii2'] < 0) or np.any(bank['amplitude'] < 0)):
        raise ValueError('Invalid feature bank or input')
    normalized = (x-bank['mean'])/bank['scale']
    distance2 = np.sum((normalized[:,None,:]-bank['centers'][None,:,:])**2, axis=2)
    radius = bank['radii2']
    support = np.where(radius[None,:] > 1e-12, np.maximum(0.,1-distance2/np.maximum(radius[None,:],1e-12)), 0.)
    return np.max(support*bank['amplitude'][None,:], axis=1)


def adjust(scores, delta):
    p, d = np.asarray(scores, np.float64), np.asarray(delta, np.float64)
    if (p.ndim != 2 or p.shape[1] != 3 or d.shape != (len(p),) or not np.isfinite(p).all()
            or not np.isfinite(d).all() or np.any(d < 0) or np.any(p < 0)
            or not np.allclose(p.sum(axis=1),1.,atol=1e-6,rtol=0)):
        raise ValueError('Require probabilities and finite nonnegative correction')
    result = p.copy()
    take = d > 0
    blue = p[take,2]*np.exp(-np.minimum(d[take],700.))
    denominator = p[take,0]+blue
    updated = np.divide((p[take,0]+p[take,2])*blue, denominator,
        out=np.zeros_like(blue), where=denominator > 0)
    # Clamp numerical roundoff so no blue probability can increase.
    updated = np.minimum(updated,p[take,2])
    result[take,2] = updated
    result[take,0] += p[take,2]-updated
    return result
