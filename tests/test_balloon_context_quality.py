import pytest
import numpy as np

from dtr.tracking import iou
from dtr.data import write_json
from scripts.prepare_balloon_quality import context_anchor_ids, context_candidates, quality_target
from scripts.calibrate_balloon_quality_gate import gate_from_oof
from scripts.train_balloon_quality import fold_ids
from scripts.evaluate_balloon_quality import quality_gated_ordered, panel_frames


def anchor(identifier=1, group="frame_1", label="blue_balloon"):
    return dict(
        id=identifier,
        source_group=group,
        label=label,
        annotation_id=identifier,
        target_box=[140, 100, 180, 140],
    )


def test_context_scales_supply_both_sides_of_localization_boundary():
    positive = anchor()
    rows = context_candidates([positive], {1})
    assert len(rows) == 4
    assert [r["context_scale"] for r in rows] == [1.0, 1.5, 2.0, 3.0]
    for row, expected in zip(rows, [1.0, 1 / 2.25, 0.25, 1 / 9], strict=True):
        assert iou(row["box"], positive["target_box"]) == pytest.approx(expected)
        target = quality_target(row, [positive], [])
        assert target["kind"] == "reviewed_balloon_overlap"
        assert target["quality"] == pytest.approx(expected)
        assert "label" not in row


def test_context_box_and_padding_are_clipped_without_duplicate_hypotheses():
    positive = dict(anchor(), target_box=[0, 0, 320, 240])
    rows = context_candidates([positive], {1})
    assert len(rows) == 1 and rows[0]["box"] == [0, 0, 320, 240]
    assert rows[0]["crop_box"] == [0, 0, 320, 240]
    assert context_candidates([positive], set()) == []


def test_context_quality_uses_best_known_instance_not_forced_anchor():
    a = anchor()
    rows = context_candidates([a], {1})
    other = dict(anchor(2), target_box=rows[1]["box"])
    # Overlapping-instance context must not be labeled low quality just because
    # it was constructed around the smaller balloon.
    target = quality_target(rows[1], [a, other], [])
    assert target["quality"] == 1 and target["anchor_id"] == 2


def test_context_selection_is_order_independent_and_source_group_bounded():
    anchors = [
        anchor(1),
        anchor(2),
        anchor(3, "frame_2"),
        anchor(4, "frame_3", "red_balloon"),
        anchor(5, "frame_4", "background"),
    ]
    ids = context_anchor_ids(anchors, 2)
    assert ids == context_anchor_ids(list(reversed(anchors)), 2)
    selected = [a for a in anchors if a["id"] in ids]
    assert len(selected) == 3
    assert len({(a["label"], a["source_group"]) for a in selected}) == 3
    assert 5 not in ids


@pytest.mark.parametrize("count", [0, 25, 1.5, True])
def test_context_selection_rejects_invalid_budget(count):
    with pytest.raises(ValueError, match="1..24"):
        context_anchor_ids([anchor()], count)


def gate_fixture():
    rows = [dict(source_group=f"train_{i}", quality=0.8) for i in range(100)]
    report = dict(
        test_evaluated=False,
        deployment_approved=False,
        examples=[
            dict(source_group=r["source_group"], target=r["quality"], oof=i / 100, fold=int(f))
            for i, (r, f) in enumerate(zip(rows, fold_ids(rows), strict=True))
        ],
    )
    return rows, report


def test_gate_calibration_uses_fixed_empirical_retention_not_development():
    rows, report = gate_fixture()
    gate = gate_from_oof(rows, report)
    assert gate["threshold"] == 0.06
    assert gate["localized_retained"] == 94
    assert gate["qualification_claim"] is False
    report["examples"][5]["oof"] = 0.06
    assert gate_from_oof(rows, report)["localized_retained"] == 95


@pytest.mark.parametrize(
    "fault", ["coverage", "source", "target", "fold", "nan", "range", "test", "groups"]
)
def test_gate_calibration_rejects_changed_or_unusable_evidence(fault):
    rows, report = gate_fixture()
    first = report["examples"][0]
    if fault == "coverage":
        report["examples"].pop()
    elif fault == "source":
        first["source_group"] = "different"
    elif fault == "target":
        first["target"] = 0.1
    elif fault == "fold":
        first["fold"] = 99
    elif fault == "nan":
        first["oof"] = float("nan")
    elif fault == "range":
        first["oof"] = 1.1
    elif fault == "test":
        report["test_evaluated"] = True
    else:
        for r, e in zip(rows, report["examples"], strict=True):
            r["source_group"] = e["source_group"] = "same"
        for e, f in zip(report["examples"], fold_ids(rows), strict=True):
            e["fold"] = int(f)
    with pytest.raises(ValueError):
        gate_from_oof(rows, report)


def test_quality_gate_restores_parts_but_not_rejected_merged_parents():
    parent = dict(
        box=[0, 0, 100, 100], accepted=True, label="blue_balloon", score=0.99, box_quality=0.2
    )
    child = dict(
        box=[10, 10, 30, 30],
        accepted=False,
        suppressed=True,
        suppression_reason="confirmed_balloon_part",
        label="blue_balloon",
        score=0.95,
        box_quality=0.9,
    )
    merged = dict(
        box=[0, 0, 200, 100],
        accepted=False,
        suppressed=True,
        suppression_reason="merged_balloon_parent",
        label="blue_balloon",
        score=0.99,
        box_quality=0.95,
    )
    result = quality_gated_ordered([parent, child, merged], 0.5)
    assert [r["accepted"] for r in result] == [False, True, False]
    assert parent["accepted"] and not child["accepted"]
    assert result[0]["suppression_reason"] == "training_calibrated_quality_gate"


@pytest.mark.parametrize("threshold", [-0.1, 1.1, np.nan])
def test_gate_rejects_invalid_threshold(threshold):
    with pytest.raises(ValueError):
        quality_gated_ordered([], threshold)


def test_reviewed_panel_requires_explicit_matching_identity(tmp_path):
    with pytest.raises(ValueError, match="explicit"):
        panel_frames("reviewed", {})
    queue = tmp_path / "queue"
    write_json(queue / "review.json", {})
    review = tmp_path / "review.json"
    write_json(review, {})
    with pytest.raises(ValueError, match="changed"):
        panel_frames("reviewed", dict(queue_sha256="bad", review_sha256="bad"), queue, review)
    with pytest.raises(ValueError, match="mode"):
        panel_frames("indoor", {}, queue, review)
