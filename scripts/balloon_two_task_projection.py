"""Two-task PCGrad: remove negative components against the original other gradient.

Reference: Yu et al., Gradient Surgery for Multi-Task Learning, arXiv:2001.06782.
With two tasks there is only one other-task ordering. All variables are shared;
sum projected task gradients, preserving the original summed-loss scale.
"""
import numpy as np


def project_numpy(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float)
    if a.ndim!=1 or b.shape!=a.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Require finite equally sized gradient vectors')
    dot=float(a@b)
    if dot>=0:
        return a+b
    return a-dot/float(b@b)*b+b-dot/float(a@a)*a


def project_tensor(a,b):
    import tensorflow as tf
    dot=tf.reduce_sum(a*b)
    negative=tf.minimum(dot,0.)
    return a-tf.math.divide_no_nan(negative,tf.reduce_sum(b*b))*b+b-tf.math.divide_no_nan(negative,tf.reduce_sum(a*a))*a
