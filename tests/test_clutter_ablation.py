from pathlib import Path

import pytest
from PIL import Image

from dtr.data import read_json, sha256, synthetic, write_json
from scripts import build_clutter_ablation as module


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    config = read_json("configs/balloon-red-blue.json")
    original = Path(synthetic(tmp_path / "base", config, per_class=1))
    base = read_json(original)
    for r in base["samples"]:
        r["source_image"] = f"{r['split']}/{r['label']}.jpg"
    write_json(original, base)
    expanded = original.with_name("expanded.json")
    write_json(expanded, dict(base, synthetic=False, training_approved=True,
                              parent_manifest_sha256=sha256(original), hard_negative_mode="none"))
    queue = tmp_path / "queue"
    queue.mkdir()
    Image.new("RGB", (24, 24), (12, 13, 14)).save(queue / "negative.png")
    negative = dict(path="negative.png", sha256=sha256(queue / "negative.png"), split="train",
                    label="background", session="training-negative", source_image="train/background.jpg")
    write_json(queue / "review.json", {})
    write_json(queue / "decision.json", {})
    monkeypatch.setattr(module, "checked_hard_negatives", lambda *args: [negative])
    # The final-test pixels must be unavailable to this development operation.
    for row in base["samples"]:
        if row["split"] == "test":
            (original.parent / row["path"]).unlink()
    return expanded, original, queue, queue / "decision.json", tmp_path / "out"


@pytest.mark.parametrize("mode", ["admit", "control"])
def test_dose_and_holdout_boundary(inputs, mode):
    result = module.build(*inputs, mode, repetitions=1)
    assert result["added_entries"] == 1 and result["counts"]["train/background"] == 2
    doc = read_json(inputs[-1] / "manifest.json")
    before = [r for r in read_json(inputs[0])["samples"] if r["split"] != "train"]
    after = [r for r in doc["samples"] if r["split"] != "train"]
    assert [{**r, "path": "base/" + r["path"]} for r in before] == after
    assert all(not (inputs[-1] / r["path"]).exists() for r in after if r["split"] == "test")
    with pytest.raises(FileExistsError):
        module.build(*inputs, mode)


@pytest.mark.parametrize("repetitions", [0, 5, True, 1.5])
def test_invalid_dose_rejected_before_io(tmp_path, repetitions):
    with pytest.raises(ValueError, match="repetitions"):
        module.build("absent", "absent", "absent", "absent", tmp_path / "out", "admit", repetitions)


def test_reserved_family_is_never_admitted(inputs):
    path = inputs[0]
    doc = read_json(path)
    next(r for r in doc["samples"] if r["split"] == "train")["source_image"] = "train/IMG_123.jpg"
    write_json(path, doc)
    with pytest.raises(ValueError, match="Reserved IMG"):
        module.build(*inputs, "admit")
    assert not inputs[-1].exists()


def test_repeated_clutter_admission_is_rejected(inputs):
    path = inputs[0]
    doc = read_json(path)
    doc["hard_negative_mode"] = "admit"
    write_json(path, doc)
    with pytest.raises(ValueError, match="previously added"):
        module.build(*inputs, "admit")


@pytest.mark.parametrize("conflict", ["positive_hash", "heldout_session"])
def test_expanded_partition_conflicts_fail_before_output(inputs, monkeypatch, conflict):
    rows = read_json(inputs[0])["samples"]
    row = dict(path="negative.png", sha256=sha256(inputs[2] / "negative.png"),
               split="train", label="background", source_image="train/background.jpg", session="train")
    if conflict == "positive_hash":
        row["sha256"] = next(r["sha256"] for r in rows if r["label"] == "red_balloon")
    else:
        row["session"] = next(r["session"] for r in rows if r["split"] == "test")
    monkeypatch.setattr(module, "checked_hard_negatives", lambda *args: [row])
    with pytest.raises(ValueError, match="conflicts"):
        module.build(*inputs, "admit")
    assert not inputs[-1].exists()
