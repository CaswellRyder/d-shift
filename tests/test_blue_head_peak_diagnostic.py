import numpy as np
import pytest

from scripts.audit_blue_head_outcomes import peak_cells


def test_peak_switch_can_change_box_with_identical_geometry():
    a = np.zeros((15,20,6),np.float32)
    a[...,:2] = -10
    a[...,2:] = [.5,.5,.2,.2]
    a[8,5,1],a[9,5,1] = 1.,.99
    b = a.copy()
    b[8,5,1],b[9,5,1] = .98,1.
    pa,pb = peak_cells(a),peak_cells(b)
    np.testing.assert_array_equal(a[...,2:],b[...,2:])
    assert len(pa) == len(pb) == 1
    assert pa[0]['cell'] == [5,8] and pb[0]['cell'] == [5,9]
    assert pa[0]['box'] != pb[0]['box']


def test_peak_diagnostic_rejects_invalid_grid():
    with pytest.raises(ValueError):
        peak_cells(np.zeros((1,1,6),np.float32))
    with pytest.raises(ValueError):
        peak_cells(np.full((15,20,6),np.nan,np.float32))
