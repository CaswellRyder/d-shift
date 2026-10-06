"""Split-policy regression tests; no downloads or private credentials."""

import importlib.util
from pathlib import Path

from PIL import Image

from dtr.data import read_json, write_json

spec = importlib.util.spec_from_file_location(
    "prepare_roboflow_dtr", Path(__file__).parents[1] / "scripts/prepare_roboflow_dtr.py"
)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)

proposal_spec = importlib.util.spec_from_file_location(
    "evaluate_dtr_proposals", Path(__file__).parents[1] / "scripts/evaluate_dtr_proposals.py"
)
proposal_evaluator = importlib.util.module_from_spec(proposal_spec)
proposal_spec.loader.exec_module(proposal_evaluator)


def test_fixed_groups_and_temporal_guards():
    assert builder.assignment("Highbay_123_jpg.rf.abc.jpg")[0] == "test"
    for time in ("192000", "192459"):
        assert builder.assignment(f"20241002_{time}_003_jpg.rf.abc.jpg")[0] == "val"
    for time in ("191900", "191959", "192500", "192559"):
        assert builder.assignment(f"20241002_{time}_003_jpg.rf.abc.jpg")[0] is None
    for time in ("191859", "192600"):
        assert builder.assignment(f"20241002_{time}_003_jpg.rf.abc.jpg")[0] == "train"
    assert builder.assignment("unknown.jpg")[0] is None
    assert builder.assignment("20241003_192000_003_jpg.rf.abc.jpg")[0] is None


def test_cross_split_hash_filter_reserves_test_without_moving_images():
    rows = [
        dict(source="a", split="train", dhash=[0] * 8),
        dict(source="b", split="val", dhash=[3] + [0] * 7),
        dict(source="c", split="test", dhash=[0] * 8),
        dict(source="d", split="train", dhash=[255] * 8),
        dict(source="e", split="train", dhash=[255] * 8),
    ]
    kept, dropped = builder.remove_cross_split_near_duplicates(rows)
    assert [r["source"] for r in kept] == ["c", "d", "e"]
    assert {r["source"] for r in dropped} == {"a", "b"}
    assert all(r["match"] == "c" for r in dropped)
    assert rows[0]["split"] == "train"


def test_duplicate_annotation_signature_ignores_order_but_not_classes():
    a = {"name": "Green Balloon", "bbox": [1, 2, 3, 4]}
    b = {"name": "Purple Balloon", "bbox": [10, 20, 30, 40]}
    assert builder.signature({"boxes": [a, b]}) == builder.signature({"boxes": [b, a]})
    assert builder.signature({"boxes": [a]}) != builder.signature({"boxes": [b]})


def test_qualification_does_not_claim_independent_sessions_or_flight_readiness():
    qualification = builder.qualification()
    assert "NOT verified recording sessions" in qualification["grouping"]
    assert not qualification["competition_accuracy_established"]
    assert not qualification["distillation_approved"]
    assert not qualification["flight_or_pi_qualified"]


def test_proposal_iou():
    assert proposal_evaluator.iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1
    assert proposal_evaluator.iou([0, 0, 10, 10], [10, 10, 20, 20]) == 0
    assert proposal_evaluator.iou([0, 0, 0, 0], [0, 0, 0, 0]) == 0
    assert proposal_evaluator.iou([0, 0, 10, 10], [0, 0, 5, 5]) == 0.25


def test_proposal_evaluation_is_validation_only_and_scales_boxes(tmp_path, monkeypatch):
    source = tmp_path / "source"
    (source / "valid").mkdir(parents=True)
    Image.new("RGB", (100, 100)).save(source / "valid/frame.jpg")
    write_json(
        source / "valid/_annotations.coco.json",
        {
            "images": [{"id": 1, "file_name": "frame.jpg", "width": 100, "height": 100}],
            "categories": [{"id": 1, "name": "Green Balloon"}, {"id": 2, "name": "Balloons"}],
            "annotations": [
                {"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20]},
                {"image_id": 1, "category_id": 2, "bbox": [1, 1, 2, 2]},
            ],
        },
    )
    monkeypatch.setattr(
        proposal_evaluator,
        "proposals",
        lambda rgb, task, **kwargs: [{"box": [32, 24, 96, 72], "area": 3072}],
    )
    output = tmp_path / "report.json"
    proposal_evaluator.evaluate(source, output)
    report = read_json(output)
    assert report["split"] == "val" and not report["test_evaluated"]
    assert not report["model_inference"]
    assert list(report["classes"]) == ["Green Balloon"]
    assert report["classes"]["Green Balloon"]["coverage_iou_0.5"] == 1
