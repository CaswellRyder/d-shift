from copy import deepcopy
import pytest

from dtr.data import read_json
from dtr.detector_metrics import summarize, match, size_bin


def standard():
    result = read_json("configs/goal-detection-standard.json")
    result["minimum_targets_per_class"] = 1
    return result


def frame():
    names = standard()["classes"]
    truth = [dict(label=k, box=[i*20, 0, i*20+10, 10]) for i, k in enumerate(names)]
    return dict(image_size=[320, 240], truth=truth,
                predictions=[dict(t, score=0.9) for t in truth])


def test_per_class_pass_is_not_deployment_approval():
    result = summarize([frame()], standard())
    assert result["development_passed"]
    assert not result["deployment_approved"] and not result["test_evaluated"]
    assert result["colors"]["orange"]["recall"] == 1


def test_duplicate_counts_as_false_positive_and_low_score_does_not_count():
    f = frame()
    f["predictions"].append(deepcopy(f["predictions"][0]))
    f["predictions"].append(dict(f["predictions"][1], score=0.1))
    result = summarize([f], standard())
    assert not result["development_passed"]
    assert result["classes"]["orange_circle"]["precision"] == 0.5
    assert result["classes"]["orange_square"]["precision"] == 1


def test_wrong_shape_does_not_pass_but_localization_records_overlap():
    f = frame()
    f["predictions"][0]["label"] = "orange_triangle"
    result = summarize([f], standard())
    assert result["class_agnostic_localization"]["recall"] == 1
    assert result["classes"]["orange_circle"]["recall"] == 0
    assert not result["development_passed"]


def test_empty_or_insufficient_evidence_cannot_pass():
    assert not summarize([], standard())["development_passed"]
    strict = standard()
    strict["minimum_targets_per_class"] = 50
    assert not summarize([frame()], strict)["development_passed"]


def test_matching_is_one_to_one_and_size_reference_is_fixed():
    target = dict(label="orange_circle", box=[0, 0, 10, 10])
    result = match([target, target], [dict(target, score=0.9)], 0.5)
    assert len(result) == 1 and result[0][1] == 0
    assert size_bin([0, 0, 30, 30], 640, 640) == "under16px"
    assert size_bin([0, 0, 64, 64], 640, 640) == "32pluspx"


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -0.01, 1.01])
def test_invalid_scores_cannot_disappear_below_confidence_threshold(score):
    f = frame()
    f["predictions"].append(dict(f["predictions"][0], score=score))
    with pytest.raises(ValueError, match="score"):
        summarize([f], standard())


@pytest.mark.parametrize("box", [[0, 0, 0, 1], [0, 0, float("nan"), 1]])
def test_invalid_boxes_reject_report(box):
    f = frame()
    f["predictions"][0]["box"] = box
    with pytest.raises(ValueError, match="box"):
        summarize([f], standard())
