from copy import deepcopy

import cv2
import numpy as np
import pytest

from dtr.vision import color_masks, suppress_duplicates


def row(box, label="yellow_circle", score=0.95, accepted=True):
    return dict(box=box, label=label, score=score, accepted=accepted)


def test_nested_rims_keep_outer_extent_not_highest_inner_score():
    original = [row([10, 10, 40, 40], score=0.85), row([15, 15, 35, 35], score=0.999)]
    saved = deepcopy(original)
    found = suppress_duplicates(original)
    assert original == saved
    assert found[0]["accepted"] and not found[1]["accepted"]
    assert found[1]["raw_accepted"] and found[1]["suppressed"]
    assert found[1]["suppressed_by"] == 0
    assert found[1]["rejection_reason"] == "duplicate"


def test_adjacent_different_class_small_or_rejected_boxes_survive():
    rows = [
        row([0, 0, 30, 30]),
        row([25, 0, 55, 30]),
        row([5, 5, 25, 25], label="orange_square"),
        row([10, 10, 15, 15]),
        row([0, 0, 30, 30], accepted=False),
    ]
    found = suppress_duplicates(rows)
    assert [r["accepted"] for r in found] == [True, True, True, True, False]
    assert not any(r["suppressed"] for r in found)


def test_overlap_uses_confidence_and_none_is_passthrough():
    rows = [row([0, 0, 30, 30], score=0.85), row([7, 0, 37, 30], score=0.99)]
    assert [r["accepted"] for r in suppress_duplicates(rows)] == [False, True]
    assert all(r["accepted"] for r in suppress_duplicates(rows, "none"))
    assert suppress_duplicates([]) == []
    with pytest.raises(ValueError):
        suppress_duplicates(rows, "unknown")


def test_suppression_references_final_surviving_candidate():
    rows = [
        row([0, 0, 30, 30], score=0.85),
        row([5, 5, 25, 25], score=0.99),
        row([7, 0, 37, 30], score=0.999),
    ]
    found = suppress_duplicates(rows)
    assert [r["accepted"] for r in found] == [False, False, True]
    assert found[0]["suppressed_by"] == found[1]["suppressed_by"] == 2


def test_expanded_orange_profile_recovers_desaturated_rims_only_when_enabled():
    rgb = cv2.cvtColor(np.uint8([[[18, 35, 160]]]), cv2.COLOR_HSV2RGB)
    assert color_masks(rgb, "goal", "v2")[0][0, 0] == 0
    assert color_masks(rgb, "goal", "orange_v3")[0][0, 0] == 255
    np.testing.assert_array_equal(
        color_masks(rgb, "balloon", "v2"), color_masks(rgb, "balloon", "orange_v3")
    )
    with pytest.raises(ValueError):
        color_masks(rgb, "goal", "invalid")
