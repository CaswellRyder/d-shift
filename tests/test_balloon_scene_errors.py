import pytest

from scripts.audit_balloon_scene_errors import analyze_frame, maximum_matching
from scripts.evaluate_red_blue_development import detection_counts


def target(box, label="blue_balloon"):
    return dict(box=box, label=label)


def detection(box, label="blue_balloon", accepted=True, raw=True, score=.99):
    return dict(box=box, label=label, accepted=accepted, raw_accepted=raw, score=score)


def frame(truth, rows):
    return dict(id=0, source="fixture", truth=truth, detections=rows,
                counts=detection_counts(truth, rows))


def test_maximum_matching_reassigns_shared_candidate():
    truth = [target([0,0,10,10]), target([4,0,14,10])]
    rows = [detection([2,0,12,10]), detection([0,0,9,10])]
    assert len(maximum_matching(truth, rows, lambda t,d: True)) == 2
    assert len(maximum_matching(truth, rows[:1], lambda t,d: True)) == 1


@pytest.mark.parametrize("row,reason", [
    (detection([20,20,30,30]), "no_localized_proposal"),
    (detection([0,0,10,10], label="background", accepted=False, raw=False), "different_predicted_class"),
    (detection([0,0,10,10], accepted=False, raw=False), "below_acceptance_threshold"),
    (detection([0,0,10,10], accepted=False), "suppressed"),
])
def test_missed_target_attribution(row, reason):
    result = analyze_frame(frame([target([0,0,10,10])], [row]))
    assert result["false_negatives"][0]["reason"] == reason


def test_duplicate_wrong_color_and_localization_are_separate():
    truth = [target([0,0,10,10])]
    rows = [detection([0,0,10,10]), detection([0,0,10,10], score=.95),
            detection([0,0,10,10], label="red_balloon"),
            detection([5,0,15,10]), detection([20,20,30,30])]
    report = analyze_frame(frame(truth, rows))
    assert {e["reason"] for e in report["false_positives"]} == {
        "duplicate", "wrong_color", "localization_overlap", "little_or_no_annotated_overlap"}
    assert report["matched_upper_bounds"]["geometry_oracle"] == 1
    assert not report["false_negatives"]


def test_oracle_does_not_double_count_one_box_for_two_targets():
    report = analyze_frame(frame([target([0,0,10,10]), target([4,0,14,10])],
                                 [detection([2,0,12,10])]))
    assert report["false_negatives"][0]["reason"] == "one_to_one_conflict"
    assert report["matched_upper_bounds"]["geometry_oracle"] == 1


def test_tampered_counts_rejected():
    example = frame([target([0,0,10,10])], [])
    example["counts"]["blue_balloon"]["fn"] = 0
    with pytest.raises(ValueError, match="Stored counts"):
        analyze_frame(example)
