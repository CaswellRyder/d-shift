import numpy as np
import pytest

from scripts.runtime_positive_weights import weights


def test_only_reviewed_wrong_positive_gets_added_weight():
    labels = np.array([1,1,2,0,0])
    teacher = np.eye(3)[[0,0,2,1,2]]
    ordinary = np.array([True,True,True,True,False])
    reviewed = np.array([True,False,True,True,False])
    actual,mask = weights(ordinary,reviewed,teacher,labels)
    np.testing.assert_array_equal(actual,[16,1,1,1,4])
    np.testing.assert_array_equal(mask,[True,False,False,False,False])


def test_reviewed_cannot_silently_reweight_unprotected_rows():
    with pytest.raises(ValueError):
        weights(np.array([False]),np.array([True]),np.array([[1.,0,0]]),np.array([1]))
    with pytest.raises(ValueError):
        weights(np.array([True]),np.array([True]),np.array([[np.nan,0,0]]),np.array([1]))
