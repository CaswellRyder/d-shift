from pathlib import Path

import pytest
from PIL import Image

from dtr.data import read_json, sha256, synthetic, write_json
from scripts import append_reviewed_balloon_views as appender
from scripts import prepare_indoor_balloon_review as preparer


@pytest.mark.parametrize("limit", [0, 25, True, 2.5])
def test_frame_limit_checked_before_io(tmp_path, limit):
    with pytest.raises(ValueError, match="candidate frames"):
        preparer.build("missing", "missing", tmp_path / "out", limit)


@pytest.mark.parametrize("invalid", ["identity", "pixels", "family", "escape"])
def test_prior_queue_must_be_matching_training_only(tmp_path, monkeypatch, invalid):
    archive = tmp_path / "dataset.zip"
    archive.write_bytes(b"fixture archive not opened")
    monkeypatch.setattr(preparer, "ARCHIVE_SHA", sha256(archive))
    manifest = tmp_path / "manifest.json"
    write_json(manifest, {})
    prior = tmp_path / "prior"
    prior.mkdir()
    Image.new("RGB", (10, 10), "blue").save(prior / "frame.png")
    frame = dict(path="frame.png", sha256=sha256(prior / "frame.png"),
                 source="frame_0012_jpg.rf.abc.jpg")
    if invalid == "pixels":
        frame["sha256"] = "changed"
    elif invalid == "family":
        frame["source"] = "IMG_1234_jpg.rf.abc.jpg"
    elif invalid == "escape":
        frame["path"] = "../outside.png"
    write_json(prior / "review.json", dict(archive_sha256=preparer.ARCHIVE_SHA,
        base_manifest_sha256="changed" if invalid == "identity" else sha256(manifest), frames=[frame]))
    with pytest.raises(ValueError):
        preparer.build(tmp_path, manifest, tmp_path / "out", 12, prior / "review.json")
    assert not (tmp_path / "out").exists()


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    config = read_json("configs/balloon-red-blue.json")
    original = Path(synthetic(tmp_path / "base", config, per_class=1))
    base = read_json(original)
    for row in base["samples"]:
        row["source_image"] = f"{row['split']}/{row['label']}.jpg"
    write_json(original, base)
    manifest = original.with_name("expanded.json")
    write_json(manifest, dict(base, synthetic=False, training_approved=True,
                             parent_manifest_sha256=sha256(original)))
    queue = tmp_path / "queue"
    queue.mkdir()
    Image.new("RGB", (20, 20), (25, 20, 200)).save(queue / "crop.png")
    extra = dict(path="crop.png", sha256=sha256(queue / "crop.png"), split="train",
                 label="blue_balloon", session="expansion-train", source_image="train/new.jpg",
                 source_group="new-group")
    write_json(queue / "review.json", dict(samples=[extra]))
    review = queue / "decision.json"
    write_json(review, dict(admit=[0], exclude=[], training_approved=True))
    monkeypatch.setattr(appender, "admitted", lambda *args: [extra])
    for row in base["samples"]:
        if row["split"] == "test":
            (original.parent / row["path"]).unlink()
    return manifest, original, queue, review, tmp_path / "out"


@pytest.mark.parametrize("control", [False, True])
def test_expansion_matches_labels_and_preserves_reserved_metadata(inputs, control):
    receipt = appender.build(*inputs, control)
    assert receipt["added_entries"] == 1
    assert receipt["counts"]["train/blue_balloon"] == 2
    before = [r for r in read_json(inputs[0])["samples"] if r["split"] != "train"]
    after = [r for r in read_json(inputs[-1] / "manifest.json")["samples"] if r["split"] != "train"]
    assert [{**r, "path": "base/" + r["path"]} for r in before] == after
    assert all(not (inputs[-1] / r["path"]).exists() for r in after if r["split"] == "test")
    with pytest.raises(FileExistsError):
        appender.build(*inputs, control)


@pytest.mark.parametrize("decision", [dict(admit=[], exclude=[]), dict(admit=[0], exclude=[0]),
    dict(admit=[True], exclude=[]), dict(admit=[0], exclude=[], training_approved=False)])
def test_incomplete_or_unsafe_review_rejected(inputs, decision):
    write_json(inputs[3], dict(training_approved=True) | decision)
    with pytest.raises(ValueError, match="complete training-only review"):
        appender.build(*inputs)
    assert not inputs[-1].exists()
