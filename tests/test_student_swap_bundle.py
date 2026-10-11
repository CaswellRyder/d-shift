import json

import pytest

from scripts import build_student_swap_pi_bundle as swap
from scripts import pi_capture_exposure_frames as capture

CONTROLS = {"ExposureTime": 40000, "AnalogueGain": 8.0, "ColourGains": (1.0, 2.3)}


def fake_frames(monkeypatch, metadata):
    frames = iter(metadata)
    monkeypatch.setattr(capture, "capture_rgb", lambda camera: (None, next(frames)))


def test_settle_waits_for_gain_and_colour_not_just_exposure(monkeypatch):
    lagging = dict(ExposureTime=40000, AnalogueGain=8.0, ColourGains=(1.2, 1.8))
    fake_frames(monkeypatch, [lagging, lagging, dict(lagging, ColourGains=(1.0, 2.3))])
    assert capture.settle(None, CONTROLS, limit=3)


def test_settle_reports_failure_when_sensor_never_matches(monkeypatch):
    fake_frames(monkeypatch, [dict(ExposureTime=40000, AnalogueGain=12.0, ColourGains=(1.0, 2.3))] * 3)
    assert not capture.settle(None, CONTROLS, limit=3)


def reference(tmp_path):
    root = tmp_path / "reference"
    root.mkdir()
    (root / "bundle.json").write_text(json.dumps({"files": {}}))
    (root / "inputs.json").write_text(json.dumps({"models": {"old": {}}, "frames": [], "crops": []}))
    (root / "search-golden.json").write_text(json.dumps({"old": {}}))
    return root


@pytest.mark.parametrize("name", ["old", "Bad Name", ""])
def test_builder_rejects_existing_or_invalid_alias_before_writing(tmp_path, name):
    output = tmp_path / "out"
    with pytest.raises(ValueError):
        swap.prepare(reference(tmp_path), output, "unused.so", name, tmp_path / "m.tflite")
    assert not output.exists()
