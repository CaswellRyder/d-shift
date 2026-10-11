import numpy as np
import pytest

from scripts.balanced_localizer_data import extra_indices, expand, run_path


def test_control_repeats_only_old_fit_rows_without_replacement():
    ids = extra_indices('control')
    assert len(ids) == len(set(ids)) == 54 and np.all((ids >= 0) & (ids < 121))
    np.testing.assert_array_equal(ids, extra_indices('control'))


def test_public_rows_repeated_exactly_three_additional_times():
    ids, counts = np.unique(extra_indices('balanced'), return_counts=True)
    np.testing.assert_array_equal(ids, np.arange(121, 139))
    np.testing.assert_array_equal(counts, np.full(18, 3))


@pytest.mark.parametrize('arm', ['control', 'balanced'])
def test_expand_preserves_every_original_row_and_exact_labels(arm):
    x, y = np.arange(139)[:, None], np.arange(139)[:, None] * 10
    a, b = expand(x, y, arm)
    assert len(a) == len(b) == 193
    np.testing.assert_array_equal(a[:139], x)
    np.testing.assert_array_equal(b[:139], y)
    np.testing.assert_array_equal(a[139:], x[extra_indices(arm)])
    np.testing.assert_array_equal(b[139:], y[extra_indices(arm)])


def test_invalid_arm_and_dataset_fail():
    with pytest.raises(ValueError):
        extra_indices('holdout')
    with pytest.raises(ValueError):
        expand(np.arange(140), np.arange(140), 'balanced')
    with pytest.raises(ValueError):
        run_path('balanced', 49)
