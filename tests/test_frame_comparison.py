import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "comparison", Path(__file__).parents[1] / "scripts/compare_dtr_frames.py"
)
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def fixture_report():
    return dict(
        split="val",
        frames=1,
        image_size=[320, 240],
        proposal_limit=12,
        annotation_sha256="same-labels",
        vision_sha256="same-proposals",
        test_evaluated=False,
        tasks={
            "balloon": dict(
                threshold=0.8,
                model_sha256="model",
                frames=[{"file": "frame"}],
                classes={"green": dict(targets=4, true_positive=2, false_positive=3)},
            )
        },
    )


def test_comparison_counts_duplicates_as_false_positives():
    before = fixture_report()
    after = copy.deepcopy(before)
    after["tasks"]["balloon"]["classes"]["green"]["false_positive"] = 1
    report = comparison.compare(before, after)
    metrics = report["tasks"]["balloon"]
    assert metrics["before"]["precision"] == 0.4
    assert metrics["after"]["recall"] == 0.5
    assert metrics["after"]["f1"] == pytest.approx(4 / 7)
    assert not report["deployment_approved"]


@pytest.mark.parametrize(
    "key,value",
    [
        ("vision_sha256", "changed"),
        ("annotation_sha256", "changed"),
        ("proposal_limit", 3),
        ("test_evaluated", True),
    ],
)
def test_scope_changes_refused(key, value):
    before = fixture_report()
    after = copy.deepcopy(before)
    after[key] = value
    with pytest.raises(ValueError):
        comparison.compare(before, after)


def test_explicit_vision_ablation_keeps_teachers_fixed():
    before = fixture_report()
    after = copy.deepcopy(before)
    after["vision_sha256"] = "new-vision"
    report = comparison.compare(before, after, allow_vision_change=True)
    assert report["vision_changed"]
    after["tasks"]["balloon"]["model_sha256"] = "new-teacher"
    with pytest.raises(ValueError, match="fixed teacher"):
        comparison.compare(before, after, allow_vision_change=True)


def test_budget_change_requires_explicit_unequal_compute_comparison():
    before = fixture_report()
    after = copy.deepcopy(before)
    after["proposal_limits"] = {"balloon": 24}
    with pytest.raises(ValueError, match="budget"):
        comparison.compare(before, after, allow_vision_change=True)
    report = comparison.compare(before, after, allow_budget_change=True)
    assert report["budget_changed"] and not report["same_proposal_budget"]
    after["tasks"]["balloon"]["model_sha256"] = "changed"
    with pytest.raises(ValueError, match="fixed teacher"):
        comparison.compare(before, after, allow_budget_change=True)
