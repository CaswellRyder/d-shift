from pathlib import Path

import keras
import numpy as np
import pytest
import tensorflow as tf
from PIL import Image

from dtr.data import batches, load_rgb, prepare_coco, read_json, synthetic, validate, write_json
from dtr.export import export_int8
from dtr.models import student, teacher
from dtr.runtime import Predictor, benchmark, quantize
from dtr.training import Distiller
from dtr.vision import proposals, replay

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    return read_json(ROOT / "configs/balloon.json")


@pytest.fixture
def dataset(tmp_path, config):
    return synthetic(tmp_path / "data", config, per_class=2)


def test_teacher_shape_and_roundtrip(tmp_path):
    model = teacher(7)
    assert model.output_shape == (None, 7)
    assert len([x for x in model.layers if x.name.endswith("residual")]) == 10
    assert 2_000_000 < model.count_params() < 3_000_000
    image = np.ones((1, 96, 96, 3), np.float32) * 128
    expected = model(image, training=False).numpy()
    model.save(tmp_path / "teacher.keras")
    loaded = keras.models.load_model(tmp_path / "teacher.keras", compile=False)
    np.testing.assert_allclose(expected, loaded(image, training=False).numpy(), atol=1e-6)


def test_dataset_is_explicitly_synthetic(dataset, config):
    doc = validate(dataset, config)
    assert doc["synthetic"] and len(doc["samples"]) == 18


def test_cross_split_session_leakage_rejected(dataset, config):
    doc = read_json(dataset)
    doc["samples"][-1]["session"] = doc["samples"][0]["session"]
    write_json(dataset, doc)
    with pytest.raises(ValueError, match="session leakage"):
        validate(dataset, config)


def test_manifest_tampering_rejected(dataset, config):
    doc = read_json(dataset)
    doc["samples"][0]["sha256"] = "bad"
    write_json(dataset, doc)
    with pytest.raises(ValueError, match="changed"):
        validate(dataset, config)


def test_student_preprocessing_matches_runtime(dataset, config):
    ds = batches(dataset, config, "train", 96, paired=True)
    x, _ = next(iter(ds))
    row = read_json(dataset)["samples"][0]
    expected = load_rgb(Path(dataset).parent / row["path"], 64)
    np.testing.assert_array_equal(x["student"][0].numpy(), expected)


def test_distillation_updates_only_student(config):
    mentor = teacher(3)
    pupil = student(3)
    distiller = Distiller(pupil, mentor, config)
    distiller.compile(optimizer=keras.optimizers.Adam(1e-3))
    x = {"teacher": tf.ones((2, 96, 96, 3)) * 120, "student": tf.ones((2, 64, 64, 3)) * 120}
    before_teacher = [v.numpy().copy() for v in mentor.weights]
    before_student = pupil.weights[0].numpy().copy()
    result = distiller.train_step((x, tf.constant([1, 2])))
    assert np.isfinite(float(result["loss"]))
    assert not np.array_equal(before_student, pupil.weights[0].numpy())
    for before, after in zip(before_teacher, mentor.weights):
        np.testing.assert_array_equal(before, after.numpy())


def test_quantization_clips_instead_of_wrapping():
    np.testing.assert_array_equal(quantize(np.array([-999, 0, 999]), 1, 0), [-128, 0, 127])
    with pytest.raises(ValueError):
        quantize(np.array([0]), 0, 0)


def test_integer_export_and_runtime(tmp_path, dataset, config):
    path = tmp_path / "student.tflite"
    receipt = export_int8(student(3), dataset, config, path, {"test": True})
    assert receipt["integer_only_verified"] and not receipt["deployment_approved"]
    with pytest.raises(ValueError, match="Unvalidated"):
        Predictor(path)
    predictor = Predictor(path, allow_unvalidated=True)
    result = predictor.predict(np.zeros((80, 80, 3), dtype=np.uint8))
    assert len(result["scores"]) == 3
    assert sum(result["scores"]) == pytest.approx(1, abs=1e-5)
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checksum"):
        Predictor(path, allow_unvalidated=True)


def test_proposals_are_bounded():
    image = np.zeros((240, 320, 3), np.uint8)
    image[20:60, 10:50] = (30, 220, 30)
    image[90:130, 150:190] = (30, 220, 30)
    candidates = proposals(image, "balloon", limit=1)
    assert len(candidates) == 1
    assert all(0 <= n <= 320 for n in candidates[0]["box"])


def test_replay_and_benchmark_require_explicit_research_mode(tmp_path, dataset, config):
    import cv2
    import json

    model = tmp_path / "student.tflite"
    export_int8(student(3), dataset, config, model, {"test": True})
    video = tmp_path / "fixture.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 5, (320, 240))
    assert writer.isOpened()
    frame = np.zeros((240, 320, 3), np.uint8)
    frame[30:100, 50:120] = (30, 220, 30)
    for _ in range(3):
        writer.write(frame)
    writer.release()
    output = tmp_path / "observations.jsonl"
    with pytest.raises(ValueError, match="Unvalidated"):
        replay(video, model, output)
    result = replay(video, model, output, max_frames=2, allow_unvalidated=True)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert result["frames"] == len(rows) == 2
    assert all(r["flight_commands"] is None and r["observations"] for r in rows)
    with pytest.raises(FileExistsError):
        replay(video, model, output, allow_unvalidated=True)
    sample = Path(dataset).parent / read_json(dataset)["samples"][0]["path"]
    with pytest.raises(ValueError, match="Unvalidated"):
        benchmark(model, sample)
    receipt = benchmark(model, sample, count=2, allow_unvalidated=True)
    assert receipt["samples"] == 2 and not receipt["pi_zero_verified"]


def test_coco_import_requires_explicit_split_policy(tmp_path, config):
    with pytest.raises(ValueError, match="sessions"):
        prepare_coco(tmp_path / "raw", tmp_path / "out", config)


def test_quarantined_coco_rejected(tmp_path, config):
    write_json(
        tmp_path / "raw/train/_annotations.coco.json",
        {"info": {"training_approved": False}, "images": [], "annotations": [], "categories": []},
    )
    with pytest.raises(ValueError, match="quarantined"):
        prepare_coco(tmp_path / "raw", tmp_path / "out", config, trust_source_splits=True)
    assert not (tmp_path / "out").exists()


def test_coco_import(tmp_path, config):
    rng = np.random.default_rng(18)
    for split in ("train", "valid", "test"):
        folder = tmp_path / "raw" / split
        folder.mkdir(parents=True)
        Image.fromarray(rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)).save(
            folder / "frame.png"
        )
        write_json(
            folder / "_annotations.coco.json",
            {
                "images": [{"id": 1, "file_name": "frame.png"}],
                "categories": [
                    {"id": 1, "name": "Green Balloon"},
                    {"id": 2, "name": "Purple Balloon"},
                ],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [5, 5, 20, 20]},
                    {"id": 2, "image_id": 1, "category_id": 2, "bbox": [150, 150, 20, 20]},
                ],
            },
        )
    manifest = prepare_coco(tmp_path / "raw", tmp_path / "out", config, trust_source_splits=True)
    doc = validate(manifest, config)
    assert not doc["synthetic"]
    assert "UNVERIFIED" in doc["split_provenance"]
