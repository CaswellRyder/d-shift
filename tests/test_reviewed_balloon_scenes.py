import numpy as np
from PIL import Image
import pytest

from dtr.data import sha256, write_json
from scripts.evaluate_reviewed_balloon_scenes import classify_candidates, load_predictor, reviewed_frames
from scripts.prepare_balloon_scene_review import development_family


def fixture(tmp_path):
    path = tmp_path / "frame.png"
    Image.fromarray(np.zeros((240, 320, 3), np.uint8)).save(path)
    row = dict(id=0, path="frame.png", sha256=sha256(path),
               provisional_truth=[dict(label="blue_balloon", box=[1, 2, 30, 40])])
    review = dict(evaluation_approved=True, training_approved=False, deployment_approved=False,
                  admit=[0], exclude=[])
    return row, review


def save(tmp_path, row, review):
    queue = tmp_path / "review.json"
    write_json(queue, dict(frames=[row]))
    review.setdefault("queue_sha256", sha256(queue))
    path = tmp_path / "decision.json"
    write_json(path, review)
    return path


def test_reviewed_frames_keep_true_empty_negatives(tmp_path):
    row, review = fixture(tmp_path)
    row["provisional_truth"] = []
    frames, _ = reviewed_frames(tmp_path, save(tmp_path, row, review))
    assert len(frames) == 1
    assert frames[0][0]["provisional_truth"] == []


def test_review_orientation_matches_pi_bundle_pixels_and_boxes(tmp_path):
    from scripts.build_development_panel_bundle import bundle
    row, review = fixture(tmp_path)
    rgb = np.zeros((240, 320, 3), np.uint8)
    rgb[2:40, 1:30, 2] = 255
    Image.fromarray(rgb).save(tmp_path / "frame.png")
    row.update(sha256=sha256(tmp_path / "frame.png"), source="scene")
    review["vertical_flip"] = [0]
    decision = save(tmp_path, row, review)
    frames, _ = reviewed_frames(tmp_path, decision)
    bundled = bundle(tmp_path, decision, tmp_path / "bundle")
    found, pixels = frames[0]
    assert found["vertical_flip"] is True
    assert found["provisional_truth"][0]["box"] == [1, 200, 30, 238]
    assert found["provisional_truth"] == bundled[0]["truth"]
    with Image.open(tmp_path / "bundle" / bundled[0]["path"]) as image:
        np.testing.assert_array_equal(pixels, np.asarray(image))
    assert row["provisional_truth"][0]["box"] == [1, 2, 30, 40]
    assert pixels.flags.c_contiguous


@pytest.mark.parametrize("flips", [[True], [1], [0, 0], "0"])
def test_reject_invalid_orientation_review(tmp_path, flips):
    row, review = fixture(tmp_path)
    review["vertical_flip"] = flips
    with pytest.raises(ValueError, match="Vertical flips"):
        reviewed_frames(tmp_path, save(tmp_path, row, review))


@pytest.mark.parametrize("change", [dict(queue_sha256="changed"), dict(evaluation_approved=False),
    dict(training_approved=True), dict(admit=[0, 0]), dict(admit=[True]),
    dict(exclude=[0]), dict(admit=[]), dict(admit=[1])])
def test_reject_incomplete_or_unsafe_review(tmp_path, change):
    row, review = fixture(tmp_path)
    review.update(change)
    with pytest.raises(ValueError):
        reviewed_frames(tmp_path, save(tmp_path, row, review))


@pytest.mark.parametrize("change", [dict(id=1), dict(path="../escape.png"),
    dict(sha256="changed"), dict(provisional_truth=[dict(label="orange_balloon",box=[1,2,3,4])]),
    dict(provisional_truth=[dict(label="red_balloon",box=[1,2,400,4])])])
def test_reject_changed_pixels_and_invalid_truth(tmp_path, change):
    row, review = fixture(tmp_path)
    row.update(change)
    with pytest.raises(ValueError):
        reviewed_frames(tmp_path, save(tmp_path, row, review))


def test_photo_family_groups_variants_and_video_frames():
    assert development_family("IMG_5110_JPG.rf.abc.jpg") == "IMG_5110"
    assert development_family("IMG_6100_00113_jpg.rf.abc.jpg") == "IMG_6100"
    assert development_family("frame_0012_jpg.rf.abc.jpg") is None


@pytest.mark.parametrize("teacher", [True, False])
def test_scene_prediction_preserves_crop_order_and_bounds_teacher_batches(teacher):
    calls = []

    class FakePredictor:
        metadata = {"kind": "keras_teacher" if teacher else "student"}

        def predict_many(self, crops):
            calls.append(len(crops))
            return [{"score": int(crop[0, 0, 0])} for crop in crops]

        def predict(self, crop):
            return self.predict_many([crop])[0]

    rgb = np.zeros((1, 130, 3), np.uint8)
    rgb[0, :, 0] = np.arange(130)
    candidates = [dict(crop_box=[i, 0, i + 1, 1], candidate_id=i) for i in range(130)]
    result = classify_candidates(FakePredictor(), rgb, candidates)
    assert [r["score"] for r in result] == list(range(130))
    assert [r["candidate_id"] for r in result] == list(range(130))
    assert calls == ([64, 64, 2] if teacher else [1] * 130)
    calls.clear()
    assert classify_candidates(FakePredictor(), rgb, []) == []
    assert calls == []


def test_scene_evaluator_rejects_unknown_model_format():
    with pytest.raises(ValueError, match="Expected"):
        load_predictor("model.unknown")


def test_unknown_selection_rejected_before_loading_frames():
    from scripts.evaluate_reviewed_balloon_scenes import evaluate
    with pytest.raises(ValueError, match="selection policy"):
        evaluate("absent", "absent", {}, selection="unknown")
