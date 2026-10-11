import numpy as np
import pytest

from scripts.source_resolved_negative_data import outside_boxes, assert_novel
from scripts.prepare_direct_hybrid_negatives import clear_of_targets


def test_nearby_chair_requires_explicit_exception_but_contains_no_target():
    truth = [{'box': [78.4, 153.96984, 106.4, 188.1408]}]
    box = [85, 196, 112, 215]
    assert outside_boxes(box, truth)
    assert not clear_of_targets(box, truth)


def test_intersecting_or_invalid_crop_cannot_be_negative():
    assert not outside_boxes([85, 180, 112, 215], [{'box': [78.4, 153, 106.4, 188.14]}])
    with pytest.raises(ValueError):
        outside_boxes([-1, 0, 10, 10], [])


def test_exact_duplicate_teacher_input_is_forbidden():
    base = np.zeros((2, 4, 4, 3), np.float32)
    new = np.ones((1, 4, 4, 3), np.float32)
    assert_novel(base, new)
    with pytest.raises(ValueError):
        assert_novel(base, base[:1])
    with pytest.raises(ValueError):
        assert_novel(base, np.repeat(new, 2, axis=0))
