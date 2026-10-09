import hashlib
import io
import zipfile

import numpy as np
from PIL import Image
import pytest

from dtr.data import sha256, write_json
from scripts import audit_balloon_pair_supervision as audit


def detection(box, label="red_balloon"):
    return dict(box=box, label=label)


def target(box, identifier=0, label="red_balloon"):
    return dict(box=box, id=identifier, label=label)


def test_same_target_supports_parent_or_child_preference():
    parent, child = detection([0, 0, 100, 100]), detection([20, 20, 40, 40])
    p, reason = audit.preference(parent, child, [target(parent["box"])])
    assert reason == "admitted_for_audit" and p["choice"] == "parent"
    p, _ = audit.preference(parent, child, [target(child["box"])])
    assert p["choice"] == "child"


def test_different_known_instances_are_not_duplicate_labels():
    parent, child = detection([0, 0, 100, 100]), detection([20, 20, 40, 40])
    assert audit.preference(parent, child, [target(parent["box"]), target(child["box"], 1)]) == (
        None,
        "different_targets",
    )


@pytest.mark.parametrize(
    "truth,reason",
    [
        ([], "unmatched"),
        ([target([200, 200, 210, 210])], "unmatched"),
        ([target([0, 0, 100, 100], label="blue_balloon")], "unmatched"),
        ([target([0, 0, 100, 100]), target([0, 0, 100, 100], 1)], "ambiguous_target"),
        ([target([0, 0, 320, 240])], "no_clear_preference"),
    ],
)
def test_unknown_or_ambiguous_pairs_not_automatic_negatives(truth, reason):
    assert audit.preference(detection([0, 0, 100, 100]), detection([20, 20, 40, 40]), truth) == (
        None,
        reason,
    )


def test_repeated_views_do_not_inflate_source_coverage():
    rows = [
        dict(choice="parent", source_group="same", domain="one", label="red_balloon")
        for _ in range(50)
    ]
    counts = audit.coverage(rows)
    assert counts["parent"]["pairs"] == 50 and counts["parent"]["source_groups"] == 1
    assert counts["parent"]["largest_source_fraction"] == 1
    assert counts["child"]["pairs"] == 0 and counts["child"]["largest_source_fraction"] is None


class FakePredictor:
    def __init__(self, accepted=True):
        self.accepted = accepted

    def predict(self, rgb):
        return dict(
            accepted=self.accepted, label="red_balloon", score=0.99, scores=[0.005, 0.99, 0.005]
        )


def test_pair_audit_requires_live_acceptance_and_preserves_proposals():
    proposals = [
        dict(box=[0, 0, 100, 100], crop_box=[0, 0, 110, 110]),
        dict(box=[20, 20, 40, 40], crop_box=[18, 18, 42, 42]),
    ]
    frame = dict(
        source="train.jpg",
        source_group="same",
        domain="fixture",
        rgb=np.zeros((240, 320, 3), np.uint8),
        proposals=proposals,
        truth=[target([0, 0, 100, 100])],
    )
    pairs, excluded = audit.frame_pairs(frame, FakePredictor())
    assert len(pairs) == 1 and pairs[0]["choice"] == "parent" and not excluded
    assert audit.frame_pairs(frame, FakePredictor(False)) == ([], {})
    assert "accepted" not in proposals[0]


@pytest.fixture
def photos(tmp_path, monkeypatch):
    image = Image.new("RGB", (320, 240), "red")
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    archive = tmp_path / "photos.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("balloon/train/allowed.png", stream.getvalue())
    # Held-out image deliberately absent: this loader must never try to read it.
    doc = dict(
        training_approved=True,
        synthetic=False,
        source_archive_sha256=sha256(archive),
        samples=[
            dict(
                split="train",
                label="red_balloon",
                source_image="balloon/train/allowed.png",
                source_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                annotation_id=1,
                box_xyxy=[10, 10, 30, 30],
            ),
            dict(
                split="test",
                label="red_balloon",
                source_image="balloon/train/reserved.png",
                source_sha256="reserved",
            ),
        ],
    )
    manifest = tmp_path / "manifest.json"
    write_json(manifest, doc)
    monkeypatch.setattr(audit, "BOOTSTRAP_SHA", sha256(manifest))
    return manifest, archive, doc


def test_only_training_pixels_opened(photos):
    manifest, archive, _ = photos
    frames = audit.training_photos(manifest, archive)
    assert len(frames) == 1 and frames[0]["source"] == "balloon/train/allowed.png"
    assert frames[0]["truth"][0]["box"] == [10, 10, 30, 30]


@pytest.mark.parametrize(
    "fault",
    [
        "manifest_hash",
        "archive_hash",
        "approval",
        "synthetic",
        "img",
        "path",
        "held_source",
        "held_hash",
        "pixel_hash",
        "conflict",
    ],
)
def test_source_guard_fails_closed(photos, monkeypatch, fault):
    manifest, archive, doc = photos
    row = doc["samples"][0]
    if fault == "archive_hash":
        doc["source_archive_sha256"] = "bad"
    elif fault == "approval":
        doc["training_approved"] = False
    elif fault == "synthetic":
        doc["synthetic"] = True
    elif fault == "img":
        row["source_image"] = "balloon/train/IMG_reserved.png"
    elif fault == "path":
        row["source_image"] = "balloon/train/../bad.png"
    elif fault == "held_source":
        doc["samples"][1]["source_image"] = row["source_image"]
    elif fault == "held_hash":
        doc["samples"][1]["source_sha256"] = row["source_sha256"]
    elif fault == "pixel_hash":
        row["source_sha256"] = "bad"
    elif fault == "conflict":
        doc["samples"].append(dict(row, label="blue_balloon"))
    else:
        doc["extra"] = "change"
    write_json(manifest, doc)
    if fault != "manifest_hash":
        monkeypatch.setattr(audit, "BOOTSTRAP_SHA", sha256(manifest))
    with pytest.raises(ValueError):
        audit.training_photos(manifest, archive)
