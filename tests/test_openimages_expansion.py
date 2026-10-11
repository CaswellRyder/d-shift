import pytest

from scripts.acquire_more_openimages_balloons import unseen_ids


def test_new_acquisition_is_disjoint_deterministic_and_bounded():
    assert unseen_ids({'c': [], 'a': [], 'b': []}, ['a'], 1) == ['b']
    assert unseen_ids({'a': []}, ['a'], 1) == []
    assert len(unseen_ids({str(i): [] for i in range(200)}, [], 96)) == 96
    for bad in (0, 97, True, 1.5):
        with pytest.raises(ValueError, match='bounded'):
            unseen_ids({}, [], bad)
