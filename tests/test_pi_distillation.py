import importlib.util
from pathlib import Path

import pytest

from dtr.data import batches, read_json, synthetic, validate, write_json
from dtr.export import export_int8
from dtr.models import student

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "distill_pi_student", ROOT / "scripts/distill_pi_student.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def scoped_data(tmp_path):
    config = read_json(ROOT / "configs/balloon.json")
    manifest = synthetic(tmp_path / "data", config, per_class=1)
    # Reserved test pixels are deliberately unavailable to the development run.
    for row in read_json(manifest)["samples"]:
        if row["split"] == "test":
            (Path(manifest).parent / row["path"]).unlink()
    return manifest, config


def test_development_validation_does_not_open_test(scoped_data):
    manifest, config = scoped_data
    validate(manifest, config, splits=("train", "val"))
    with pytest.raises(FileNotFoundError):
        validate(manifest, config)
    ds = batches(manifest, config, "val", 64, validation_splits=("train", "val"))
    assert sum(len(y) for _, y in ds) == 3
    with pytest.raises(ValueError, match="outside validated scope"):
        batches(manifest, config, "test", 64, validation_splits=("train", "val"))


def test_test_metadata_still_detects_leakage(scoped_data):
    manifest, config = scoped_data
    doc = read_json(manifest)
    doc["samples"][-1]["sha256"] = doc["samples"][0]["sha256"]
    write_json(manifest, doc)
    with pytest.raises(ValueError, match="image leakage"):
        validate(manifest, config, splits=("train", "val"))


def test_missing_test_checksum_rejected(scoped_data):
    manifest, config = scoped_data
    doc = read_json(manifest)
    del doc["samples"][-1]["sha256"]
    write_json(manifest, doc)
    with pytest.raises(ValueError, match="checksum"):
        validate(manifest, config, splits=("train", "val"))


def test_scoped_export_never_needs_test_images(scoped_data, tmp_path):
    manifest, config = scoped_data
    receipt = export_int8(student(3), manifest, config, tmp_path / "model.tflite",
                          {"test_fixture": True}, validation_splits=("train", "val"))
    assert receipt["integer_only_verified"]
    assert receipt["calibration_split"] == "train"
    assert not receipt["pi_zero_verified"]


@pytest.mark.parametrize("logs", [{}, {"loss": float("nan")}, {"loss": float("inf")}])
def test_nonfinite_training_refuses_export(logs):
    with pytest.raises(ValueError, match="refusing export"):
        module.require_finite_metrics(logs)


def test_finite_training_metrics():
    module.require_finite_metrics({"loss": 1.2, "val_hard_loss": 0.7})
