from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from dtr.data import read_json, sha256, synthetic, write_json
from scripts.prepare_balloon_instance_review import checked_rows
from scripts.prepare_balloon_instance_review import prepare, unmatched_acceptances
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


def test_bad_candidate_scope_rejected_before_loading_data():
    with pytest.raises(ValueError, match="candidate scope"):
        prepare(*(["absent"] * 7), candidate_scope="unknown")


def test_bad_proposal_variant_rejected_before_loading_data():
    with pytest.raises(ValueError, match="proposal variant"):
        prepare(*(["absent"] * 7), variant="unknown")


def test_mining_requires_all_candidates_before_io():
    with pytest.raises(ValueError, match="all candidate"):
        prepare(*(["absent"] * 7), mining_model="missing.tflite")


def test_mining_preserves_unlabeled_status_and_skips_annotated_targets():
    calls = []

    class FakePredictor:
        def predict(self, crop):
            calls.append(crop.shape)
            return dict(label="blue_balloon", score=.9, scores=[.1,0.,.9], accepted=True,
                        latency_ms=999)

    rgb = np.zeros((240,320,3), np.uint8)
    boxes = [[0,0,20,20], [100,100,130,130]]
    candidates = [dict(box=b, crop_box=b) for b in boxes]
    rows = unmatched_acceptances(rgb, candidates, [dict(box=boxes[0])], FakePredictor())
    assert calls == [(30,30,3)]
    assert len(rows) == 1 and rows[0]["box"] == boxes[1]
    assert "label" not in rows[0] and "mining_prediction" not in candidates[1]
    assert rows[0]["mining_prediction"] == dict(label="blue_balloon", score=.9, scores=[.1,0.,.9], accepted=True)


@pytest.mark.parametrize("label,accepted", [("background", True), ("blue_balloon", False)])
def test_mining_skips_rejected_or_background_predictions(label, accepted):
    class FakePredictor:
        def predict(self, crop):
            return dict(label=label, accepted=accepted)

    rgb = np.zeros((240,320,3), np.uint8)
    assert unmatched_acceptances(rgb, [dict(box=[0,0,10,10], crop_box=[0,0,10,10])], [], FakePredictor()) == []


@pytest.mark.parametrize("scope,count", [("added", 1), ("all", 2)])
@pytest.mark.parametrize("variant", ["mser_instance_split", "mser_blue_cores", "lab_blue_split"])
def test_expansion_can_skip_old_sources_and_select_all_candidates(inputs, tmp_path, monkeypatch, scope, count, variant):
    from scripts import prepare_balloon_instance_review as module
    manifest, queue, review = inputs
    archive = tmp_path/"unused.zip"
    archive.write_bytes(b"fixture")
    monkeypatch.setattr(module, "original_frames", lambda *a: pytest.fail("Original sources must be skipped"))
    pixels = np.full((240, 320, 3), 20, np.uint8)
    monkeypatch.setattr(module, "indoor_frames", lambda *a: [("background.jpg", pixels, [])])
    proposals = [dict(box=b, crop_box=b, color_group=1) for b in ([0,0,10,10], [20,20,30,30])]
    calls = []

    def search(rgb, search_variant):
        calls.append(search_variant)
        return proposals[:1] if search_variant == "mser" else proposals

    monkeypatch.setattr(module, "experimental", search)
    if variant == "lab_blue_split":
        from scripts import research_balloon_color_split
        monkeypatch.setattr(research_balloon_color_split, "proposals", lambda rgb: search(rgb,variant))
    out = tmp_path/"review-expansion"
    result = prepare(manifest, manifest, archive, queue, review, archive, out,
                     indoor_only=True, candidate_scope=scope, variant=variant)
    assert result["crops"] == count
    generated = read_json(out/"review.json")
    assert generated["candidate_scope"] == scope and generated["indoor_only"]
    assert generated["proposal_variant"] == variant and calls == ["mser", variant]
    assert generated["color_split_sha256"] == (sha256("scripts/research_balloon_color_split.py")
                                               if variant == "lab_blue_split" else None)
    assert all(r["label"] == "unreviewed" for r in generated["samples"])


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
