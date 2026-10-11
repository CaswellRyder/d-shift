import copy

import pytest

from scripts.calibrate_balloon_split_geometry import calibrate
from scripts.compare_calibrated_balloon_selection import calibrated_selection, raw_candidates
from scripts.pi_balloon_search_bench import select_detections


def training(box, source="train/approved.jpg"):
    return dict(panel=[(source, None, [dict(annotation_id=1, label="blue_balloon", box=box)])])


def candidate(box):
    return dict(box=box, crop_box=box, label="blue_balloon", score=.95, scores=[.02,.03,.95], accepted=True)


def test_training_shape_quantile_is_bounded_by_existing_candidate_limits():
    assert calibrate(training([0,0,20,10]))["max_aspect"] == 2.
    assert calibrate(training([0,0,10,10]))["max_aspect"] == 1.5
    assert calibrate(training([0,0,100,10]))["max_aspect"] == 3.


@pytest.mark.parametrize("bad", ["train/IMG_1.jpg", "train/img_2.png"])
def test_reserved_development_family_cannot_calibrate_shape(bad):
    with pytest.raises(ValueError, match="Reserved IMG"):
        calibrate(training([0,0,20,10], bad))


def test_calibration_rejects_invalid_or_duplicate_labels():
    with pytest.raises(ValueError, match="Invalid"):
        calibrate(training([0,0,0,10]))
    data = training([0,0,20,10])
    data["panel"].append(data["panel"][0])
    with pytest.raises(ValueError, match="Duplicate"):
        calibrate(data)


def test_calibrated_bound_keeps_two_oval_children_instead_of_merged_parent():
    rows = [candidate([0,0,100,100]), candidate([0,0,80,40]), candidate([0,60,80,100])]
    saved = copy.deepcopy(rows)
    control = calibrated_selection(rows, 1.5)
    changed = calibrated_selection(rows, 2.)
    assert [r["accepted"] for r in control] == [True, False, False]
    assert [r["accepted"] for r in changed] == [False, True, True]
    assert rows == saved
    assert [r["score"] for r in changed] == [r["score"] for r in rows]


def test_training_control_is_identical_and_raw_reconstruction_is_lossless():
    rows = [candidate([0,0,100,100]), candidate([0,0,80,40]), candidate([0,60,80,100])]
    selected = select_detections(rows, "split_merged")
    assert calibrated_selection(rows, 1.5) == selected
    assert raw_candidates(selected) == rows


@pytest.mark.parametrize("value", [1., 3.1, float("inf"), float("nan")])
def test_invalid_calibration_bound_fails_closed(value):
    with pytest.raises(ValueError, match="aspect bound"):
        calibrated_selection([], value)


def test_missing_raw_acceptance_is_not_guessed():
    with pytest.raises(ValueError, match="Raw acceptance"):
        raw_candidates([candidate([0,0,10,10])])
