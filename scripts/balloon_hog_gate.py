"""Fixed low-cost shape features and source-isolated weighted ridge fitting."""
import hashlib

import cv2
import numpy as np

from scripts.audit_public_balloon_export import original_image_key


def features(images):
    hog = cv2.HOGDescriptor((64,64),(32,32),(16,16),(16,16),9,1,-1,0,.2,False,64,False)
    values = []
    for image in images:
        x = np.asarray(image)
        if x.shape != (64,64,3) or not np.isfinite(x).all() or np.any(x < 0) or np.any(x > 255):
            raise ValueError('Require canonical64x64 RGB pixels')
        if not np.array_equal(x, x.astype(np.uint8)):
            raise ValueError('Do not silently quantize noninteger image inputs')
        values.append(hog.compute(cv2.cvtColor(x.astype(np.uint8),cv2.COLOR_RGB2GRAY)).ravel())
    result = np.asarray(values,np.float64)
    if result.shape != (len(images),324) or not np.isfinite(result).all():
        raise ValueError('Unexpected HOG output')
    return result


def source_folds(rows):
    parents = list(range(len(rows)))

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    seen = {}
    aliases = [original_image_key(r['source_image']) for r in rows]
    for i,row in enumerate(rows):
        for key in (('source',aliases[i]),('pixels',row['sha256'])):
            if key in seen:
                parents[root(i)] = root(seen[key])
            seen[key] = i
    groups = {}
    for i,name in enumerate(aliases):
        groups.setdefault(root(i),set()).add(name)
    names = {k:'|'.join(sorted(v)) for k,v in groups.items()}
    ids = [names[root(i)] for i in range(len(rows))]
    folds = np.array([int(hashlib.sha256(name.encode()).hexdigest(),16)%5 for name in ids])
    return ids, folds


def fit(x, labels):
    x, labels = np.asarray(x,np.float64), np.asarray(labels)
    if (x.ndim != 2 or labels.shape != (len(x),) or not np.isfinite(x).all()
            or not np.isin(labels,[0,1,2]).all() or not np.any(labels>0) or not np.any(labels==0)):
        raise ValueError('Require finite features and both foreground/background')
    y = (labels>0).astype(np.float64)
    weights = np.where(y>0,.5/y.sum(),.5/(len(y)-y.sum()))
    mean = np.sum(x*weights[:,None],axis=0)
    scale = np.maximum(np.sqrt(np.sum((x-mean)**2*weights[:,None],axis=0)),.001)
    z = (x-mean)/scale
    coefficients = np.linalg.solve(z.T@(weights[:,None]*z)+.1*np.eye(x.shape[1]), z.T@(weights*(y-.5)))
    scores = z@coefficients+.5
    return dict(mean=mean,scale=scale,coefficients=coefficients,intercept=np.array(.5),
        threshold=np.array(scores[labels>0].min()))


def score(x, model):
    x = np.asarray(x,np.float64)
    if x.ndim != 2 or x.shape[1:] != model['mean'].shape or not np.isfinite(x).all():
        raise ValueError('Invalid shape features')
    return (x-model['mean'])/model['scale']@model['coefficients']+model['intercept']


def fold_fit(x, labels, folds, withheld):
    keep = np.asarray(folds) != withheld
    if keep.all() or not keep.any():
        raise ValueError('Require disjoint nonempty fit/withheld partitions')
    model = fit(x[keep], labels[keep])
    return model, score(x[~keep],model), keep
