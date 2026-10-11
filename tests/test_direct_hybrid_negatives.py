import pytest

from scripts.prepare_direct_hybrid_negatives import clear_of_targets


TRUTH = [dict(box=[100, 100, 120, 120])]


def test_disjoint_crops_are_eligible():
    assert clear_of_targets([0, 0, 40, 40], TRUTH)
    assert clear_of_targets([0, 0, 320, 240], [])


def test_balloon_parts_and_guard_overlap_excluded():
    assert not clear_of_targets([100, 100, 105, 105], TRUTH)
    assert not clear_of_targets([80, 80, 90, 90], TRUTH)
    assert not clear_of_targets([0, 0, 320, 240], TRUTH)


def test_exact_guard_boundary_and_invalid_geometry():
    assert clear_of_targets([0, 0, 84, 84], TRUTH)
    for box in ([0, 0, 0, 0], [-1, 0, 20, 20], [0, 0, 321, 240], [0, 0, float('nan'), 10]):
        with pytest.raises(ValueError):
            clear_of_targets(box, TRUTH)
    with pytest.raises(ValueError):
        clear_of_targets([0, 0, 10, 10], TRUTH, guard=0)
