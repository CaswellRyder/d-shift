import copy

import pytest

from scripts.calibrate_balloon_split_geometry import calibrate, calibration_bound
from scripts.compare_calibrated_balloon_selection import calibrated_selection, raw_candidates
from scripts.pi_balloon_search_bench import golden_key, select_detections
from scripts.summarize_balloon_search_pi import group_key, verify_golden_policy


def training(box, source="train/approved.jpg"):
    return dict(panel=[(source, None, [dict(annotation_id=1, label="blue_balloon", box=box)])])


def candidate(box):
    return dict(box=box, crop_box=box, label="blue_balloon", score=.95, scores=[.02,.03,.95], accepted=True)


def test_training_shape_quantile_is_bounded_by_existing_candidate_limits():
    assert calibrate(training([0,0,20,10]))["max_aspect"] == 2.
    assert calibrate(training([0,0,10,10]))["max_aspect"] == 1.5
    assert calibrate(training([0,0,100,10]))["max_aspect"] == 3.
    assert calibration_bound(calibrate(training([0,0,20,10]))) == 2.


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


def test_pi_goldens_and_trial_groups_bind_the_calibrated_aspect():
    baseline = dict(model="student", search="mser_direct", selection="split_merged")
    changed = dict(baseline, split_max_aspect=2.4)
    assert group_key(baseline) != group_key(changed)
    assert golden_key("mser_direct", "split_merged") == "mser_split_merged"
    assert golden_key("mser_direct", "split_merged", 2.4) == "mser_split_merged_aspect_2.4"
    with pytest.raises(ValueError, match="selection policy"):
        verify_golden_policy(dict(changed, golden_search="mser_split_merged"), [], {})
    with pytest.raises(ValueError, match="aspect bound"):
        select_detections([], "parts", 2.4)


@pytest.mark.parametrize("fault", ["bound", "quantile", "count", "sources", "reserved", "duplicate", "nonfinite", "approval"])
def test_saved_calibration_must_reproduce_training_only_recipe(fault):
    result = calibrate(training([0,0,20,10]))
    if fault == "bound":
        result["max_aspect"] = 2.4
    elif fault == "quantile":
        result["quantile"] = .99
    elif fault == "count":
        result["annotation_count"] = 2
    elif fault == "sources":
        result["source_count"] = 2
    elif fault == "reserved":
        result["annotations"][0]["source"] = "train/IMG_heldout.jpg"
    elif fault == "duplicate":
        result["annotations"].append(result["annotations"][0])
        result["annotation_count"] = 2
    elif fault == "nonfinite":
        result["annotations"][0]["aspect_ratio"] = float("nan")
    else:
        result["deployment_approved"] = True
    with pytest.raises(ValueError):
        calibration_bound(result)


def test_pi_processing_does_not_add_neural_calls_for_calibrated_selection(monkeypatch):
    import numpy as np
    from scripts import pi_balloon_search_bench as bench
    boxes = ([0,0,100,100], [0,0,80,40], [0,60,80,100])
    monkeypatch.setattr(bench, "experimental", lambda *a: [dict(box=b, crop_box=b) for b in boxes])
    calls = []

    class Predictor:
        def predict(self, rgb):
            calls.append(rgb.shape)
            return dict(label="blue_balloon", score=.95, scores=[.02,.03,.95], accepted=True)

    rgb = np.zeros((240,320,3), np.uint8)
    control = bench.process_frame(rgb, Predictor(), "mser", selection="split_merged")
    changed = bench.process_frame(rgb, Predictor(), "mser", selection="split_merged", split_max_aspect=2.)
    assert control["neural_calls"] == changed["neural_calls"] == 3 and len(calls) == 6
    assert [r["scores"] for r in control["detections"]] == [r["scores"] for r in changed["detections"]]
    assert [r["accepted"] for r in control["detections"]] == [True,False,False]
    assert [r["accepted"] for r in changed["detections"]] == [False,True,True]
