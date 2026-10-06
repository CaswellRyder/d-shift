from dtr.diagnostics import diagnose_frame


def pred(label="circle", accepted=True, **kwargs):
    return dict(box=[0, 0, 10, 10], label=label, score=0.9, accepted=accepted, **kwargs)


def test_stage_partition_and_precedence():
    truth = [dict(box=[0, 0, 10, 10], label="circle")]
    cases = [
        ([], [], "localization"),
        ([], [pred()], "proposal_budget"),
        ([pred("background", False)], [], "background"),
        ([pred("square")], [], "wrong_class"),
        ([pred("circle", False), pred("square")], [], "below_threshold"),
        ([pred("circle", False, raw_accepted=True)], [], "suppressed"),
        ([pred()], [], "detected"),
    ]
    for observations, extra, cause in cases:
        result = diagnose_frame(truth, observations, extra)
        assert result["targets"][0]["cause"] == cause


def test_duplicate_annotations_do_not_manufacture_two_detections():
    truth = [dict(box=[0, 0, 10, 10], label="circle")] * 2
    result = diagnose_frame(truth, [pred()])
    assert [r["cause"] for r in result["targets"]] == ["detected", "assignment_conflict"]


def test_false_positive_partition_and_empty_truth():
    truth = [dict(box=[0, 0, 10, 10], label="circle")]
    result = diagnose_frame(truth, [pred(), pred(), pred("square")])
    assert [r["cause"] for r in result["predictions"]] == [
        "matched", "duplicate_or_assignment", "wrong_class"
    ]
    assert diagnose_frame([], [pred()])["predictions"][0]["cause"] == "no_labeled_overlap"


def test_partial_box_is_not_localization_success():
    p = pred()
    p["box"] = [0, 0, 4, 4]
    result = diagnose_frame([dict(box=[0, 0, 10, 10], label="circle")], [p])
    assert result["targets"][0]["cause"] == "localization"
    assert result["predictions"][0]["cause"] == "partial_box"
