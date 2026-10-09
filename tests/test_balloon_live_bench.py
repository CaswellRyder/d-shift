import copy
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.pi_balloon_live_bench import (
    capture_rgb,
    frame_timing,
    run_camera,
    validate_camera_config,
)


class Request:
    def __init__(self, stamp=100, shape=(240, 320, 3)):
        self.pixels = np.zeros(shape, np.uint8)
        self.pixels[..., 0] = 200  # Blue in camera RGB888/BGR memory.
        self.stamp = stamp
        self.released = False

    def make_array(self, name):
        assert name == "main"
        return self.pixels

    def get_metadata(self):
        return {"SensorTimestamp": self.stamp}

    def release(self):
        self.released = True
        self.pixels[:] = 0


class Camera:
    def __init__(self, requests):
        self.requests = iter(requests)

    def capture_request(self, wait):
        assert wait is False
        return next(self.requests)

    def wait(self, job, timeout):
        assert timeout == 3.0
        return job


def test_rgb_is_owned_and_order_correct_after_request_release():
    request = Request()
    rgb, meta = capture_rgb(Camera([request]))
    assert tuple(rgb[0, 0]) == (0, 0, 200)
    assert meta["SensorTimestamp"] == 100
    assert request.released
    assert not np.shares_memory(rgb, request.pixels)


def test_bad_camera_buffer_is_released():
    request = Request(shape=(120, 160, 3))
    with pytest.raises(ValueError, match="buffer"):
        capture_rgb(Camera([request]))
    assert request.released


def test_timeout_is_propagated_without_retry():
    class BrokenCamera(Camera):
        def wait(self, job, timeout):
            raise TimeoutError("no camera frame")

    with pytest.raises(TimeoutError):
        capture_rgb(BrokenCamera([Request()]))


@pytest.mark.parametrize("sensor", [None, -1, 0, True, 201, 100.0, "100"])
def test_bad_sensor_timestamp_never_becomes_valid_age(sensor):
    row = frame_timing({"SensorTimestamp": sensor}, 150, 200, 300)
    assert not row["timestamp_valid"]
    assert row["sensor_to_result_ms"] is None
    assert not row["sensor_timestamp_increasing"]


def test_sensor_age_and_repeated_frame_detection():
    row = frame_timing({"SensorTimestamp": 1000000000}, 1010000000, 1020000000, 1300000000)
    assert row["timestamp_valid"] and row["sensor_timestamp_increasing"]
    assert row["sensor_to_result_ms"] == 300
    assert row["sensor_to_acquire_ms"] == 20
    for previous in (1000000000, 1000000001):
        assert not frame_timing(
            {"SensorTimestamp": 1000000000}, 1010000000, 1020000000, 1300000000, previous
        )["sensor_timestamp_increasing"]
    with pytest.raises(ValueError, match="boottime"):
        frame_timing({}, 10, 9, 11)


def test_live_loop_is_bounded_keeps_first_last_and_does_not_save(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    requests = [Request(100), Request(400)]
    calls = []

    def infer(rgb):
        calls.append(tuple(rgb[0, 0]))
        return dict(detections=[], processing_ms=1.0)

    clock = iter([150, 200, 300, 450, 500, 600])
    rows, samples, elapsed = run_camera(Camera(requests), infer, 2, 3, lambda: next(clock))
    assert len(rows) == len(calls) == len(samples) == 2
    assert elapsed > 0
    assert all(r.released for r in requests)
    assert all(row["sensor_timestamp_increasing"] for row in rows)
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(ValueError, match="frames"):
        run_camera(None, None, 1000, 3)


def config():
    return dict(
        main=dict(size=(320, 240), format="RGB888"),
        sensor=dict(output_size=(2592, 1944), bit_depth=10),
        queue=False,
        transform=SimpleNamespace(hflip=True, vflip=True, transpose=False),
    )


def test_event_samples_are_bounded_and_last_frame_is_retained():
    requests = [Request(i * 100 + 1) for i in range(12)]
    clocks = iter(v for i in range(12) for v in (i * 100 + 2, i * 100 + 3, i * 100 + 4))

    def infer(rgb):
        return dict(detections=[dict(accepted=True)], processing_ms=1.0)

    rows, samples, _ = run_camera(Camera(requests), infer, 12, 3, lambda: next(clocks))
    assert len(rows) == 12
    assert [i for i, _, _ in samples] == list(range(8)) + [11]
    assert all(r.released for r in requests)


def test_full_sensor_and_rotation_contract():
    validate_camera_config(config())
    for section, key, value in (
        ("main", "size", (640, 480)),
        ("main", "format", "BGR888"),
        ("sensor", "output_size", (1296, 972)),
        ("sensor", "bit_depth", 8),
    ):
        bad = copy.deepcopy(config())
        bad[section][key] = value
        with pytest.raises(ValueError, match="Camera"):
            validate_camera_config(bad)
    for key in ("hflip", "vflip", "transpose"):
        bad = config()
        setattr(bad["transform"], key, not getattr(bad["transform"], key))
        with pytest.raises(ValueError, match="Camera"):
            validate_camera_config(bad)
    bad = config()
    bad["queue"] = True
    with pytest.raises(ValueError, match="Camera"):
        validate_camera_config(bad)
