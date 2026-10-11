"""Summarize gradient directions without treating conflict as impossibility proof."""
import itertools

import numpy as np


def summarize(gradients, counts):
    if (not gradients or set(gradients)!=set(counts) or any(type(n) is not int or n<=0 for n in counts.values())
            or any(np.asarray(g).ndim!=1 or not np.isfinite(g).all() for g in gradients.values())
            or len({np.asarray(g).shape for g in gradients.values()})!=1):
        raise ValueError('Require finite, equally sized gradients and positive counts')
    n=sum(counts.values())
    contributions={k:np.asarray(g,float)*counts[k]/n for k,g in gradients.items()}
    norms={k:float(np.linalg.norm(g)) for k,g in contributions.items()}
    total=sum(contributions.values())
    pairs={}
    for a,b in itertools.combinations(gradients,2):
        ga,gb=np.asarray(gradients[a],float),np.asarray(gradients[b],float)
        denom=np.linalg.norm(ga)*np.linalg.norm(gb)
        pairs[a+'/'+b]=dict(cosine=float(np.clip(ga@gb/denom,-1,1)) if denom else None,
            # Derivative of loss A along normalized steepest descent for B.
            directional_derivative=float(-ga@gb/np.linalg.norm(gb)) if np.linalg.norm(gb) else None)
    total_norm=float(np.linalg.norm(total))
    return dict(pairs=pairs,contribution_norms=norms,total_gradient_norm=total_norm,
        cancellation_ratio=total_norm/sum(norms.values()) if sum(norms.values()) else None)
