from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from dtr.data import read_json, sha256, write_json
from scripts import train_balloon_quality as trainer
from scripts.prepare_balloon_quality import quality_target
from scripts.evaluate_balloon_quality import QualityPredictor, quality_ordered


def test_unmatched_proposals_are_unknown_not_negative():
    candidate = dict(box=[10, 10, 30, 30], crop_box=[8, 8, 32, 32])
    assert quality_target(candidate, [], []) is None
    assert quality_target(candidate, [], [dict(id=1, crop_box=[9, 9, 32, 32])]) is None
    assert quality_target(candidate, [], [dict(id=1, crop_box=candidate["crop_box"])]) == dict(
        quality=0., anchor_id=1, kind="exact_reviewed_background")
    positive = dict(id=2, target_box=[10, 10, 30, 30])
    assert quality_target(candidate, [positive], [dict(id=1, crop_box=candidate["crop_box"])]) == dict(
        quality=1., anchor_id=2, kind="reviewed_balloon_overlap")


def test_quality_is_best_single_balloon_not_union():
    candidate = dict(box=[0, 0, 100, 50], crop_box=[0, 0, 110, 60])
    truth = [dict(id=0, target_box=[0, 0, 50, 50]), dict(id=1, target_box=[50, 0, 100, 50])]
    assert quality_target(candidate, truth, [])["quality"] == .5


def test_ridge_folds_normalization_into_single_affine_head():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(30, 4)).astype(np.float32)
    x[:, 3] = 7  # Constant features cannot produce NaNs.
    y = np.clip(.5 + .1*x[:, 0] - .1*x[:, 1], 0, 1)
    w, b = trainer.fit_ridge(x, y, 1.)
    assert w.shape == (4,) and w.dtype == np.float32 and np.isfinite(b)
    assert np.abs(x@w + b-y).mean() < .02
    assert w[3] == 0
    assert np.array_equal(w, trainer.fit_ridge(x, y, 1.)[0])


@pytest.mark.parametrize("fault", ["nan", "penalty", "target", "shape", "empty"])
def test_invalid_regression_inputs_fail(fault):
    x, y, penalty = np.ones((5, 3)), np.full(5, .5), 10.
    if fault == "nan":
        x[0, 0] = np.nan
    elif fault == "penalty":
        penalty = 0
    elif fault == "target":
        y[0] = 1.2
    elif fault == "shape":
        y = y[:, None]
    else:
        x, y = x[:0], y[:0]
    with pytest.raises(ValueError):
        trainer.fit_ridge(x, y, penalty)


def test_folds_keep_source_group_together_without_random_state():
    rows = [dict(source_group="a"), dict(source_group="b"), dict(source_group="a")]
    folds = trainer.fold_ids(rows)
    assert folds[0] == folds[2] and np.array_equal(folds, trainer.fold_ids(rows))


def test_localization_quality_can_beat_higher_class_confidence_without_mutation():
    parent = dict(box=[0, 0, 100, 100], label="red_balloon", score=.999,
                  accepted=True, box_quality=.2)
    child = dict(box=[10, 10, 30, 30], label="red_balloon", score=.9,
                 accepted=False, suppressed=True, suppression_reason="confirmed_balloon_part", box_quality=.95)
    found = quality_ordered([parent, child])
    assert [r["accepted"] for r in found] == [False, True]
    assert parent["accepted"] and not child["accepted"]
    child.update(suppression_reason="ordinary_nms")
    assert [r["accepted"] for r in quality_ordered([parent, child])] == [True, False]


def test_quality_does_not_merge_different_colors_or_neighbors():
    rows = [dict(box=[0, 0, 100, 100], label="red_balloon", score=.9, accepted=True, box_quality=.2),
            dict(box=[10, 10, 30, 30], label="blue_balloon", score=.99, accepted=True, box_quality=.95),
            dict(box=[101, 0, 121, 20], label="red_balloon", score=.99, accepted=True, box_quality=.95)]
    assert all(r["accepted"] for r in quality_ordered(rows))
    rows[0]["box_quality"] = np.nan
    with pytest.raises(ValueError, match="quality"):
        quality_ordered(rows)


def test_wrong_quality_contract_rejected_before_tensorflow(tmp_path):
    path = tmp_path / "fake.tflite"
    path.write_bytes(b"not a model")
    write_json(path.with_suffix(".json"), dict(kind="regular_classifier"))
    with pytest.raises(ValueError, match="identity"):
        QualityPredictor(path)


@pytest.mark.parametrize("rgb", [np.zeros((4, 4), np.uint8), np.zeros((4, 4, 3), np.float32),
                              np.zeros((0, 4, 3), np.uint8)])
def test_quality_predictor_rejects_bad_input_before_inference(rgb):
    predictor = QualityPredictor.__new__(QualityPredictor)
    with pytest.raises(ValueError, match="RGB"):
        predictor.predict(rgb)


@pytest.fixture
def quality_config(tmp_path, monkeypatch):
    manifest = tmp_path / "original.json"
    # No holdout image file exists: only metadata may be opened.
    write_json(manifest, dict(samples=[dict(split="test", source_image="test/reserved.jpg")]))
    parent = tmp_path / "parent"
    write_json(parent / "review.json", {})
    parent_review = tmp_path / "parent-review.json"
    write_json(parent_review, {})
    root = tmp_path / "quality"
    root.mkdir()
    Image.new("RGB", (320, 240), "blue").save(root / "frame.png")
    Image.new("RGB", (24, 24), "blue").save(root / "crop.png")
    anchor = dict(id=0, source_image="train/frame_1234_jpg.rf.abc.jpg", source_sha256="source-hash",
        session="engdes2:indoor-frame-4-digits", frame_id=0, label="blue_balloon", crop_box=[8, 8, 32, 32])
    monkeypatch.setattr(trainer, "admitted", lambda *args: [anchor])
    row = dict(anchor, anchor_id=0, kind="reviewed_balloon_overlap", split="train",
        path="crop.png", sha256=sha256(root / "crop.png"), box=[10, 10, 30, 30],
        target_boxes=[[10, 10, 30, 30]], quality=1., source_group="frame_1234_jpg.jpg")
    frame = dict(id=0, path="frame.png", sha256=sha256(root / "frame.png"),
        source="frame_1234_jpg.rf.abc.jpg", crop_ids=[0],
        quality_targets=[dict(label="blue_balloon", box=[10, 10, 30, 30])])
    queue = dict(samples=[row], frames=[frame], archive_sha256=trainer.ARCHIVE_SHA,
        base_manifest_sha256=sha256(manifest), parent_queue_sha256=sha256(parent / "review.json"),
        parent_review_sha256=sha256(parent_review))
    write_json(root / "review.json", queue)
    decision = tmp_path / "decision.json"
    write_json(decision, dict(queue_sha256=sha256(root / "review.json"), admit=[0], exclude=[],
        training_approved=True, deployment_approved=False))
    return dict(manifest=str(manifest), training_approved=True, deployment_approved=False,
        sources=[dict(queue=str(root), review=str(decision), parent_queue=str(parent), parent_review=str(parent_review))])


def test_training_quality_preserves_missing_holdout_pixels(quality_config):
    rows, receipts = trainer.training_rows(quality_config)
    assert len(rows) == len(receipts) == 1 and rows[0]["quality"] == 1.


@pytest.mark.parametrize("fault", ["approval", "incomplete", "bool_id", "pixels", "frame_pixels",
    "source", "img", "split", "geometry", "target", "path", "negative", "parent", "archive"])
def test_quality_guard_rejects_corruption(quality_config, fault):
    source = quality_config["sources"][0]
    path, review_path = Path(source["queue"]) / "review.json", Path(source["review"])
    queue, review = read_json(path), read_json(review_path)
    row = queue["samples"][0]
    if fault == "approval":
        review["training_approved"] = False
    elif fault == "incomplete":
        review["admit"] = []
    elif fault == "bool_id":
        review["admit"] = [False]
    elif fault == "pixels":
        row["sha256"] = "wrong"
    elif fault == "frame_pixels":
        queue["frames"][0]["sha256"] = "wrong"
    elif fault == "source":
        row["source_image"] = "test/reserved.jpg"
    elif fault == "img":
        row["source_image"] = "train/IMG_1234.jpg"
    elif fault == "split":
        row["split"] = "val"
    elif fault == "geometry":
        row["box"][2] = 321
    elif fault == "target":
        row["quality"] = .5
    elif fault == "path":
        row["path"] = "../escape.png"
    elif fault == "negative":
        row.update(kind="exact_reviewed_background", quality=0.)
    elif fault == "parent":
        queue["parent_review_sha256"] = "wrong"
    else:
        queue["archive_sha256"] = "wrong"
    write_json(path, queue)
    review["queue_sha256"] = sha256(path)
    write_json(review_path, review)
    with pytest.raises(ValueError):
        trainer.training_rows(quality_config)
