import numpy as np
import pytest

from scripts.audit_blue_prototype_transfer import fold_bank


def test_entire_background_source_excluded_from_scaling_and_centers():
    f = np.array([[0.,0.],[10.,10.],[10.,10.],[20.,20.]])
    bank, keep, keep_centers = fold_bank(f,np.array([1,0,0,0]),['P','A','A','B'],
        f[[1,3]],np.zeros((2,3)),['A','B'],'A')
    assert keep.tolist() == [True,False,False,True]
    assert keep_centers.tolist() == [False,True]
    np.testing.assert_array_equal(bank['mean'],[10.,10.])
    np.testing.assert_array_equal(bank['scale'],[10.,10.])
    np.testing.assert_array_equal(bank['centers'],[[1.,1.]])


def test_cannot_claim_source_holdout_without_removing_it():
    with pytest.raises(ValueError,match='removed source'):
        fold_bank(np.zeros((2,2)),np.array([1,0]),['P','A'],
            np.zeros((1,2)),np.zeros((1,3)),['A'],'B')
