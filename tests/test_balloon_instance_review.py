from pathlib import Path

import pytest
from PIL import Image

from dtr.data import read_json, sha256, synthetic, write_json
from scripts.prepare_balloon_instance_review import checked_rows
from scripts.build_balloon_instance_ablation import build


@pytest.fixture
def inputs(tmp_path):
    config = read_json("configs/balloon-red-blue.json")
    manifest = Path(synthetic(tmp_path / "base", config, per_class=1))
    base = read_json(manifest)
    for i, r in enumerate(base["samples"]):
        r.update(source_image=f"{r['split']}/{r['label']}.jpg", source_sha256=str(i))
    base.update(synthetic=False, training_approved=True)
    write_json(manifest, base)
    parent = next(r for r in base["samples"] if r["split"] == "train")
    queue = tmp_path / "queue"
    queue.mkdir()
    Image.new("RGB", (320, 240), (12, 13, 14)).save(queue / "frame.png")
    Image.new("RGB", (10, 10), (12, 13, 14)).save(queue / "crop.png")
    row = dict(id=0, path="crop.png", frame_id=0, split="train", sha256=sha256(queue/"crop.png"),
               crop_box=[0, 0, 10, 10], **{k: parent[k] for k in ("source_image", "session", "source_sha256")})
    frame = dict(path="frame.png", sha256=sha256(queue/"frame.png"), source_image=parent["source_image"])
    write_json(queue/"review.json", dict(samples=[row], frames=[frame], base_manifest_sha256=sha256(manifest)))
    decision = queue/"decision.json"
    write_json(decision, dict(queue_sha256=sha256(queue/"review.json"), training_approved=True,
                              deployment_approved=False, labels=[dict(id=0, label="background")], exclude=[]))
    for r in base["samples"]:
        if r["split"] == "test":
            (manifest.parent / r["path"]).unlink()
    return manifest, queue, decision


def test_admission_requires_source_crop_parity_and_keeps_review_labels(inputs):
    rows = checked_rows(*inputs)
    assert len(rows) == 1 and rows[0]["label"] == "background"


@pytest.mark.parametrize("fault", ["hash", "missing", "duplicate", "approval", "reserved", "source", "crop", "escape", "bounds"])
def test_unsafe_review_or_crop_rejected(inputs, fault):
    manifest, queue, decision = inputs
    doc, review = read_json(queue/"review.json"), read_json(decision)
    if fault == "hash":
        review["queue_sha256"] = "wrong"
    elif fault == "missing":
        review["labels"] = []
    elif fault == "duplicate":
        review["exclude"] = [0]
    elif fault == "approval":
        review["training_approved"] = False
    else:
        row = doc["samples"][0]
        if fault == "reserved":
            row["source_image"] = "train/IMG_123.jpg"
        elif fault == "source":
            row["source_sha256"] = "unadmitted"
        elif fault == "crop":
            Image.new("RGB", (10, 10), (1, 2, 3)).save(queue/"crop.png")
            row["sha256"] = sha256(queue/"crop.png")
        elif fault == "escape":
            row["path"] = "../outside.png"
        else:
            row["crop_box"] = [-1, 0, 10, 10]
        write_json(queue/"review.json", doc)
        review["queue_sha256"] = sha256(queue/"review.json")
    write_json(decision, review)
    with pytest.raises(ValueError):
        checked_rows(manifest, queue, decision)


def test_paired_manifests_match_counts_and_never_copy_test_pixels(inputs, tmp_path):
    receipts = [build(*inputs, tmp_path/mode, mode) for mode in ("admit", "control")]
    assert receipts[0]["counts"] == receipts[1]["counts"]
    assert receipts[0]["entries_added"] == receipts[1]["entries_added"] == 4
    for mode in ("admit", "control"):
        doc = read_json(tmp_path/mode/"manifest.json")
        marked = [r for r in doc["samples"] if "instance_ablation_mode" in r]
        assert len(marked) == 4 and all(r["split"] == "train" for r in marked)
        assert all(not (tmp_path/mode/r["path"]).exists() for r in doc["samples"] if r["split"] == "test")
    with pytest.raises(FileExistsError):
        build(*inputs, tmp_path/"admit", "admit")
