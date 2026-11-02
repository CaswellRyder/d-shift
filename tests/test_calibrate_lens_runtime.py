"""The lab's saved raw frames already have the runtime camera rotation."""

import json
import sys

import cv2
import numpy as np
import pytest

from dtr.lens import LensModel
from scripts import calibrate_lens as calibration


def runtime_boards(folder):
    """Asymmetric synthetic boards make an accidental second flip observable."""
    images = []
    for index in range(10):
        image = np.full((240, 320), 180, np.uint8)
        image[0, 0], image[-1, -1] = 37, 222
        for y in range(6):
            for x in range(8):
                top, left = 40 + y * 20, 25 + index + x * 20
                image[top:top + 20, left:left + 20] = 255 if (x + y) % 2 else 0
        assert cv2.imwrite(str(folder / f"runtime-board-{index:02d}.png"), image)
        images.append(image)
    return images


def test_runtime_raw_boards_are_not_reflipped_and_produce_loadable_calibration(
        tmp_path, monkeypatch):
    folder = tmp_path / "runtime-raw"
    folder.mkdir()
    expected = runtime_boards(folder)
    seen = []
    original_detector = cv2.findChessboardCorners

    def detect(image, dimensions, flags):
        seen.append(image.copy())
        return original_detector(image, dimensions, flags)

    def forbid_second_flip(*args, **kwargs):
        raise AssertionError("Runtime frames were flipped a second time")

    def fit(objects, points, size, *args, **kwargs):
        assert len(objects) == len(points) == 10
        assert size == (320, 240)
        lens = LensModel.ideal_equidistant()
        return .12, lens.K, lens.D, [], []

    monkeypatch.setattr(cv2, "findChessboardCorners", detect)
    monkeypatch.setattr(cv2, "flip", forbid_second_flip)
    monkeypatch.setattr(cv2.fisheye, "calibrate", fit)
    output = tmp_path / "lens-runtime.json"
    result = calibration.calibrate(folder, 7, 5, .04, output,
                                   flip=False, input_flipped=True)
    assert len(seen) == 10
    assert all(np.array_equal(actual, original) for actual, original in zip(seen, expected))
    assert result["flipped"] is True and result["input_flipped"] is True
    assert result["images_used"] == 10
    assert json.loads(output.read_text()) == result
    loaded = LensModel.from_calibration(output)
    assert loaded.calibrated
    assert loaded.width == 320 and loaded.height == 240


def test_already_rotated_runtime_input_refuses_an_explicit_second_flip(tmp_path):
    output = tmp_path / "invalid-lens.json"
    with pytest.raises(ValueError, match="second time"):
        calibration.calibrate(tmp_path, 7, 5, .04, output,
                              flip=True, input_flipped=True)
    assert not output.exists()


def test_cli_runtime_input_flag_disables_default_flip(tmp_path, monkeypatch):
    captured = {}

    def capture_options(*args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(calibration, "calibrate", capture_options)
    monkeypatch.setattr(sys, "argv", ["calibrate_lens.py", str(tmp_path),
                                      "--cols", "7", "--rows", "5", "--square-size", ".04",
                                      "--output", str(tmp_path / "lens.json"), "--input-flipped"])
    calibration.main()
    assert captured["flip"] is False
    assert captured["input_flipped"] is True
