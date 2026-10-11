import numpy as np
import pytest

from scripts.cached_balloon_decoder import decode
from scripts.independent_neighbor_decoder import decode as neighbor
from scripts.research_blue_peak_blend import decode as frozen


@pytest.mark.parametrize('rescue', [False, True])
def test_random_grids_exactly_match_reference(rescue):
    rng = np.random.default_rng(413)
    for index in range(40):
        values = rng.random((15, 20, 6), dtype=np.float32)
        values[..., :2] = rng.normal(-3, 2, (15, 20, 2))
        if index % 4 == 0:
            values[..., :2] = np.round(values[..., :2])  # plateau/tie ordering
        if index % 3 == 0:
            values[..., 4:] *= .08  # small, distinct neighboring boxes
        saved = values.copy()
        reference = neighbor(values) if rescue else frozen(values, True)
        assert decode(values, rescue) == reference
        np.testing.assert_array_equal(values, saved)


@pytest.mark.parametrize('rescue', [False, True])
def test_empty_and_boundary_values(rescue):
    values = np.zeros((15, 20, 6), np.float32)
    values[..., :2] = -80
    assert decode(values, rescue) == []
    values[..., :2] = 80
    values[..., 2:] = 1
    assert decode(values, rescue) == (neighbor(values) if rescue else frozen(values, True))


def test_invalid_grid_fails_closed():
    with pytest.raises(ValueError):
        decode(np.zeros((2, 2, 6), np.float32))
    values = np.zeros((15, 20, 6), np.float32)
    values[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        decode(values)
