import io
import json
import sys
import zipfile

from PIL import Image
import pytest

from dtr.data import read_json, sha256, write_json
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


def test_summary_never_counts_replay_repetitions_as_independent_targets(tmp_path, monkeypatch):
    from scripts.summarize_red_blue_pi import main
    inputs = tmp_path / "inputs.json"
    truth = dict(label="red_balloon", box=[0, 0, 20, 20])
    write_json(inputs, dict(frames=[dict(source="one-photo", truth=[truth])]))
    results = tmp_path / "results"
    results.mkdir()
    report = results / "replay.json"
    write_json(report, dict(machine="armv6l", input_sha256=sha256(inputs), frames=3,
                            mode="replay", timing=dict(mean_ms=10)))
    frame = dict(source="one-photo", observations=[dict(**truth, score=.9, accepted=True)])
    report.with_suffix(".frames.jsonl").write_text((json.dumps(frame)+"\n")*3)
    output = tmp_path / "summary.json"
    monkeypatch.setattr(sys, "argv", ["summarize", "--results", str(results),
                                      "--inputs", str(inputs), "--output", str(output)])
    main()
    summary = read_json(output)
    assert summary["flight_ready"] is False
    result = summary["results"]["replay.json"]
    assert result["unique_development_photos"] == 1
    assert result["development_metrics"]["red_balloon"]["tp"] == 1
