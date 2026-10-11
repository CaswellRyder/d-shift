import numpy as np
import pytest

from scripts.audit_balloon_hog_gate import unique_indices


def test_duplicates_preserve_distinct_views():
    rows = [dict(sha256='a', view='original'), dict(sha256='a', view='original'),
            dict(sha256='a', view='horizontal_flip')]
    assert unique_indices(rows, np.array([2, 2, 2]), np.zeros((3, 2))).tolist() == [0, 2]


def test_conflicting_duplicate_labels_or_pixels_are_rejected():
    rows = [dict(sha256='a', view='original')]*2
    for labels, pixels in (([1, 2], [[0], [0]]), ([2, 2], [[0], [1]])):
        with pytest.raises(ValueError, match='conflicting'):
            unique_indices(rows, np.asarray(labels), np.asarray(pixels))
