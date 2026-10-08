import io
import zipfile

from PIL import Image
import pytest

from dtr.data import sha256, write_json
from scripts.pi_red_blue_bench import statistics, verify_bundle
from scripts.build_red_blue_pi_bundle import build


def test_bundle_rejects_changed_or_escaping_source(tmp_path):
    path = tmp_path / "source.py"
    path.write_text("original")
    write_json(tmp_path / "bundle.json", {"files": {"source.py": sha256(path)}})
    assert verify_bundle(tmp_path)["files"]
    path.write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        verify_bundle(tmp_path)
    write_json(tmp_path / "bundle.json", {"files": {"../elsewhere": "0"*64}})
    with pytest.raises(ValueError, match="escaped"):
        verify_bundle(tmp_path)


def test_timing_summary_preserves_bad_measurements_as_failure():
    assert statistics([1, 2, 3])["mean_ms"] == 2
    for invalid in ([], [float("nan")], [-1], [float("inf")]):
        with pytest.raises(ValueError):
            statistics(invalid)


def test_bundle_never_opens_reserved_test_images(tmp_path, monkeypatch):
    class FakePredictor:
        metadata = dict(task="balloon", classes=["background", "red_balloon", "blue_balloon"])

        def __init__(self, *args, **kwargs):
            pass

        def predict(self, path):
            return dict(scores=[0., 1., 0.], label="red_balloon", accepted=True)

    monkeypatch.setattr("scripts.build_red_blue_pi_bundle.Predictor", FakePredictor)
    crop = tmp_path / "crop.png"
    Image.new("RGB", (16, 16), "red").save(crop)
    image_bytes = io.BytesIO()
    Image.new("RGB", (32, 32), "red").save(image_bytes, format="PNG")
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as dest:
        dest.writestr("scene.png", image_bytes.getvalue())
    manifest = tmp_path / "manifest.json"
    write_json(manifest, dict(source_archive_sha256=sha256(archive), samples=[
        dict(split="test", path="DO_NOT_OPEN.png", source_image="test.png"),
        dict(split="train", path="DO_NOT_OPEN_EITHER.png", source_image="train.png"),
        dict(split="val", source_image="scene.png", path="crop.png", sha256=sha256(crop),
             label="red_balloon", annotation_id=1, box_xyxy=[0, 0, 16, 16])]))
    review = tmp_path / "review.json"
    write_json(review, dict(manifest_sha256=sha256(manifest), sources={"scene.png": {"target_ids": [1]}}))
    model = tmp_path / "model.tflite"
    model.write_bytes(b"test-model")
    write_json(model.with_suffix(".json"), FakePredictor.metadata)
    output = tmp_path / "bundle"
    result = build(output, manifest, review, archive, {"test": model})
    assert result["crops"] == 1 and result["frames"] == 1
    verify_bundle(output)
    with pytest.raises(FileExistsError):
        build(output, manifest, review, archive, {"test": model})
