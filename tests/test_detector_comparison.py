from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

from dtr.data import read_json

spec = importlib.util.spec_from_file_location(
    "compare_detector", Path(__file__).parents[1] / "scripts/compare_goal_checkpoints.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def report():
    standard = read_json("configs/goal-detection-standard.json")
    standard["minimum_targets_per_class"] = 1
    truth = [dict(label=label, box=[i*20, 0, i*20+10, 10])
             for i, label in enumerate(standard["classes"])]
    return dict(split="val", standard=standard, detector_input_width=640,
                annotation_sha256="fixture", model_sha256="a", epoch=1,
                frames=[dict(file="frame.png", image_size=[320, 240], truth=truth,
                             predictions=[dict(t, score=.9) for t in truth])],
                metrics=dict(test_evaluated=False))


def test_whole_checkpoint_ranking_uses_weakest_class_not_supplied_metrics():
    a = report()
    a["frames"][0]["predictions"].pop()
    b = report()
    b.update(model_sha256="b", epoch=2)
    for p in list(b["frames"][0]["predictions"]):
        b["frames"][0]["predictions"].append(dict(p, score=.8))
    result = module.compare([("a", a), ("b", b)])
    assert result["ranking"][0]["epoch"] == 2  # All classes .5 precision beats one zero-recall class.
    assert not result["all_requirements_met_by_top_checkpoint"]
    assert not result["promotion_allowed"]


@pytest.mark.parametrize("change", ["standard", "size", "truth", "experiment", "duplicate"])
def test_incomparable_reports_are_rejected(change):
    a = report()
    b = deepcopy(a)
    b["model_sha256"] = "b"
    if change == "standard":
        b["standard"]["confidence_threshold"] = .5
    elif change == "size":
        b["detector_input_width"] = 320
    elif change == "truth":
        b["frames"][0]["truth"].pop()
    elif change == "experiment":
        b["experiment"] = "verifier_cascade"
    else:
        b["frames"].append(deepcopy(b["frames"][0]))
    with pytest.raises(ValueError):
        module.compare([("a", a), ("b", b)])
