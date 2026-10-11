import numpy as np
import pytest

from scripts.audit_balloon_dense_proposals import decode as original
from scripts.research_blue_peak_blend import decode


def grid():
    v = np.zeros((15,20,6),np.float32)
    v[...,:2] = -10
    v[...,2:] = [.5,.5,.2,.4]
    return v


def test_control_exact_reference_parity_on_random_and_tied_grids():
    rng = np.random.default_rng(47)
    for _ in range(20):
        v = rng.uniform(0,1,(15,20,6)).astype(np.float32)
        v[...,:2] = rng.normal(-2,2,(15,20,2))
        assert decode(v,False) == original(v)
    v = grid()
    v[8:10,5,1] = 1.
    assert decode(v,False) == original(v)


def test_near_tie_blend_reduces_peak_switch_box_displacement_without_mutation():
    a = grid()
    a[8,5,1],a[9,5,1] = 1.,.99
    b = a.copy()
    b[8,5,1],b[9,5,1] = .99,1.
    saved = a.copy()
    old = np.max(np.abs(np.asarray(decode(a)[0]['box'])-decode(b)[0]['box']))
    new = np.max(np.abs(np.asarray(decode(a,True)[0]['box'])-decode(b,True)[0]['box']))
    assert old == 16. and new < .1
    assert len(decode(a,True)) == len(decode(a)) == 1
    assert decode(a,True)[0]['proposal_score'] == decode(a)[0]['proposal_score']
    np.testing.assert_array_equal(a,saved)


def test_separate_or_weak_neighbors_and_red_peaks_do_not_blend():
    a = grid()
    a[8,5,0],a[9,5,0] = 1.,.99
    assert decode(a,True) == original(a)
    a = grid()
    a[...,4:] = [.025,.025]
    a[8,5,1],a[9,5,1] = 1.,.99
    assert decode(a,True) == original(a)
    a = grid()
    a[8,5,1],a[9,5,1] = 1.,-1.
    assert decode(a,True) == original(a)
    a[11,5,1] = 1.
    assert len(decode(a,True)) == 2


def test_clipped_bounds_and_invalid_inputs():
    a = grid()
    a[0,0,1],a[0,1,1] = 1.,.99
    for d in decode(a,True):
        x,y,c,b = d['crop_box']
        assert 0 <= x < c <= 320 and 0 <= y < b <= 240
    with pytest.raises(ValueError):
        decode(np.full((15,20,6),np.nan,np.float32),True)
    a[0,0,2] = 2.
    with pytest.raises(ValueError):
        decode(a,True)
