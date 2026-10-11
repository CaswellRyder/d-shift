"""Training weights for explicit reviewed-positive teacher conflicts only."""
import numpy as np


def weights(ordinary,reviewed,teacher,labels,hard_weight=4.,positive_weight=16.):
    ordinary,reviewed = np.asarray(ordinary),np.asarray(reviewed)
    teacher,labels = np.asarray(teacher),np.asarray(labels)
    if (ordinary.dtype!=bool or reviewed.dtype!=bool or ordinary.shape!=labels.shape
            or reviewed.shape!=labels.shape or teacher.shape!=(len(labels),3)
            or not np.isfinite(teacher).all() or not np.isin(labels,[0,1,2]).all()
            or not np.isfinite([hard_weight,positive_weight]).all()
            or min(hard_weight,positive_weight)<1 or np.any(reviewed & ~ordinary)):
        raise ValueError('Invalid reviewed training weights')
    conflicts = reviewed & (labels>0) & (teacher.argmax(1)!=labels)
    result = np.where(ordinary,1.,hard_weight).astype(np.float32)
    result[conflicts] = positive_weight
    return result,conflicts
