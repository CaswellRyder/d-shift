import io
import json
import zipfile

import pytest
from PIL import Image

from dtr.data import read_json, sha256, validate
from scripts import build_public_balloon as builder
from scripts.audit_balloon_seed import audit, coverage


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    classes = ["background", "red_balloon", "blue_balloon"]
    config = dict(task="balloon", classes=classes)
    archive_path = tmp_path / "source.zip"
    index_path = tmp_path / "index.json"
    review_path = tmp_path / "review.json"
    rows = []
    with zipfile.ZipFile(archive_path, "w") as archive:
        for i, color in enumerate(((0, 255, 0), (255, 0, 0), (0, 0, 255))):
            stream = io.BytesIO()
            Image.new("RGB", (24, 24), color).save(stream, format="PNG")
            source = f"balloon/train/{i}.png"
            archive.writestr(source, stream.getvalue())
            rows.append(dict(id=i, source=source, box_xyxy=[0, 0, 24, 24]))
    index_path.write_text(json.dumps(rows))
    review_path.write_text(json.dumps(dict(source_url="fixture", **{
        label: [i] for i, label in enumerate(classes)
    })))
    monkeypatch.setattr(builder, "ARCHIVE_SHA", sha256(archive_path))
    monkeypatch.setattr(builder, "INDEX_SHA", sha256(index_path))
    return archive_path, review_path, index_path, tmp_path / "output", config


def test_training_only_seed_has_no_manufactured_holdout(inputs):
    path = builder.build(*inputs, training_only=True)
    doc = validate(path, inputs[-1], splits=("train",))
    assert {r["split"] for r in doc["samples"]} == {"train"}
    assert doc["qualification"]["training_only"]
    assert not doc["qualification"]["competition_accuracy_established"]
    assert not doc["qualification"]["redistribution_approved"]
    with pytest.raises(ValueError, match="No examples for val"):
        validate(path, inputs[-1])
    with pytest.raises(FileExistsError):
        builder.build(*inputs, training_only=True)


def test_seed_rejects_validation_source_before_writing(inputs, monkeypatch):
    index_path = inputs[2]
    rows = read_json(index_path)
    rows[0]["source"] = "balloon/val/0.png"
    index_path.write_text(json.dumps(rows))
    monkeypatch.setattr(builder, "INDEX_SHA", sha256(index_path))
    with pytest.raises(ValueError, match="upstream validation"):
        builder.build(*inputs, training_only=True)
    assert not inputs[3].exists()


def test_seed_rejects_duplicate_review_ids(inputs):
    review_path = inputs[1]
    review = read_json(review_path)
    review["red_balloon"] = review["blue_balloon"]
    review_path.write_text(json.dumps(review))
    with pytest.raises(ValueError, match="Duplicate reviewed"):
        builder.build(*inputs, training_only=True)
    assert not inputs[3].exists()


def test_seed_rejects_changed_source_archive(inputs):
    inputs[0].write_bytes(b"changed")
    with pytest.raises(ValueError, match="archive checksum"):
        builder.build(*inputs, training_only=True)


def test_coverage_requires_correct_group_and_overlap():
    box = [10, 10, 30, 30]
    assert coverage(box, "red_balloon", [dict(box=box, color_group=0)])["covered"]
    assert not coverage(box, "red_balloon", [dict(box=box, color_group=1)])["covered"]
    assert not coverage(box, "blue_balloon", [dict(box=[0, 0, 2, 2], color_group=1)])["covered"]


def test_audit_reports_coverage_without_claiming_detection_accuracy(inputs):
    manifest = builder.build(*inputs, training_only=True)
    report = audit(manifest, inputs[0])
    assert report["counts"] == {"red_balloon": dict(covered=0, total=1),
                                "blue_balloon": dict(covered=0, total=1)}
    # Fixtures fill the frame; the normal giant-proposal filter must stay enabled.
    assert not report["classification_measured"] and not report["pi_timing_measured"]
    assert not report["precision_measured"] and not report["deployment_approved"]
