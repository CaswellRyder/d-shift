import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "proposal_gate", Path(__file__).parents[1] / "scripts/compare_proposal_audits.py"
)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def report():
    counts = {f"{c} {s} Goal": dict(targets=10, covered12=5) for c in ("Orange", "Yellow")
              for s in ("Circle", "Square", "Triangle")}
    return dict(split="train", task="goal", goal_colors="all", image_size=[320, 240],
                proposal_limit=12, iou_threshold=0.5, annotation_sha256="same", frames=["one"],
                test_evaluated=False, counts=counts,
                size_counts={k + "/under16px": dict(v) for k, v in counts.items()})


def change(r, label, delta):
    r["counts"][label]["covered12"] += delta
    r["size_counts"][label + "/under16px"]["covered12"] += delta


def test_gate_requires_improvement_and_preserves_each_color_and_shape():
    before = report()
    assert not gate.compare(before, before)["eligible_for_validation"]
    after = copy.deepcopy(before)
    change(after, "Orange Circle Goal", 2)
    assert gate.compare(before, after)["eligible_for_validation"]
    change(after, "Orange Triangle Goal", -1)
    result = gate.compare(before, after)
    assert result["orange_coverage_gain"] == 1
    assert not result["eligible_for_validation"]
    assert len(result["regressions"]) == 2


def test_yellow_regression_blocks_promotion():
    before, after = report(), report()
    change(after, "Orange Circle Goal", 2)
    change(after, "Yellow Triangle Goal", -1)
    assert not gate.compare(before, after)["eligible_for_validation"]


def test_small_target_regression_cannot_hide_inside_class_gain():
    before = report()
    label = "Orange Circle Goal"
    before["counts"][label] = dict(targets=20, covered12=10)
    before["size_counts"][label + "/32pluspx"] = dict(targets=10, covered12=5)
    after = copy.deepcopy(before)
    after["counts"][label]["covered12"] = 11
    after["size_counts"][label + "/under16px"]["covered12"] = 4
    after["size_counts"][label + "/32pluspx"]["covered12"] = 7
    result = gate.compare(before, after)
    assert result["orange_coverage_gain"] == 1
    assert not result["eligible_for_validation"]


@pytest.mark.parametrize("key,value", [
    ("frames", ["another"]), ("image_size", [640, 480]), ("proposal_limit", 24),
    ("annotation_sha256", "different"), ("test_evaluated", True), ("goal_colors", "orange"),
])
def test_scope_changes_rejected(key, value):
    before, after = report(), report()
    after[key] = value
    with pytest.raises(ValueError):
        gate.compare(before, after)


def test_size_counts_must_reconcile():
    before, after = report(), report()
    after["size_counts"]["Orange Circle Goal/under16px"]["targets"] += 1
    with pytest.raises(ValueError):
        gate.compare(before, after)
