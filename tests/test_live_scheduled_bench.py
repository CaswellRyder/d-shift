import copy
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import pytest

from dtr.data import sha256, write_json
from dtr.research_search_schedule import SearchSchedule
from scripts import pi_balloon_schedule_bench as schedule_bench
from scripts import review_live_scheduled_search as reviewer
from scripts.pi_balloon_live_bench import run_camera
from scripts.pi_balloon_live_schedule_bench import (
    freshness,
    full_observation_ages,
    trial_summary,
)
from scripts.review_scheduled_balloon_search import validate_schedule

ROOT = Path(__file__).resolve().parents[1]

MS = 1_000_000
T0 = 10_000 * MS


class Request:
    def __init__(self, stamp):
        self.stamp = stamp

    def make_array(self, name):
        return np.zeros((240, 320, 3), np.uint8)

    def get_metadata(self):
        return {"SensorTimestamp": self.stamp}

    def release(self):
        pass


class Camera:
    def __init__(self, stamps):
        self.requests = iter(Request(s) for s in stamps)

    def capture_request(self, wait):
        return next(self.requests)

    def wait(self, job, timeout):
        return job


def observations(*args):
    return dict(
        detections=[],
        processing_ms=1.0,
        search_ms=0.5,
        inference_ms=0.4,
        selection_ms=0.1,
        neural_calls=0,
    )


def live_rows(monkeypatch, stamps, policy="periodic"):
    """Every clock read advances 50 ms: start, acquire, begin, finish, result per frame."""
    monkeypatch.setattr(schedule_bench, "process_frame", observations)
    ticks = iter(range(T0, T0 + 1000 * 50 * MS, 50 * MS))

    def clock():
        return next(ticks)

    scheduler = SearchSchedule()

    def infer(rgb):
        return schedule_bench.scheduled_frame(rgb, None, None, scheduler, policy, clock)

    rows, _, _ = run_camera(Camera(stamps), infer, len(stamps), 3, clock)
    return rows


def frame_stamps(count):
    # Exposure 100 ms before each capture starts; always earlier than acquisition.
    return [T0 + 5 * i * 50 * MS - 100 * MS for i in range(count)]


def test_live_scheduler_shares_camera_clock_and_ages_full_observation(monkeypatch):
    rows = live_rows(monkeypatch, frame_stamps(5))
    assert [r["mode"] for r in rows] == ["mser_direct", "mser_bright"] * 2 + ["mser_direct"]
    assert [r["reason"] for r in rows[::2]] == [
        "startup_or_failure",
        "elapsed_budget",
        "elapsed_budget",
    ]
    assert validate_schedule("periodic", rows)["mode_counts"] == dict(mser_direct=3, mser_bright=2)
    result = freshness(rows)
    # A bright frame's own exposure is fresh, but the newest full observation is older.
    assert [r["sensor_to_result_ms"] for r in rows] == [300.0] * 5
    assert result["full_observation_age_ms"] == [300.0, 550.0, 300.0, 550.0, 300.0]
    assert result["full_result_intervals_ms"] == [500.0, 500.0]
    assert result["full_sensor_intervals_ms"] == [500.0, 500.0]
    assert result["unknown_full_observation_age_frames"] == 0
    assert result["full_observation_age"]["max_ms"] == 550.0


def test_invalid_full_timestamp_never_borrows_older_observation():
    def row(mode, sensor, result, valid=True):
        return dict(
            mode=mode,
            sensor_timestamp_ns=sensor,
            timestamp_valid=valid,
            result_boottime_ns=result,
        )

    rows = [
        row("mser_bright", 5, 10),
        row("mser_direct", 10, 20),
        row("mser_bright", 30, 40),
        row("mser_direct", None, 60, valid=False),
        row("mser_bright", 70, 80),
        row("mser_direct", 90, 100),
    ]
    assert full_observation_ages(rows) == [None, 1e-5, 3e-5, None, None, 1e-5]


def test_scheduler_time_outside_camera_frame_fails(monkeypatch):
    rows = live_rows(monkeypatch, frame_stamps(3))
    for field, value in (
        ("start_ns", rows[1]["capture_completed_boottime_ns"] - 1),
        ("end_ns", rows[1]["result_boottime_ns"] + 1),
    ):
        broken = copy.deepcopy(rows)
        broken[1][field] = value
        with pytest.raises(ValueError, match="outside"):
            freshness(broken)


def test_unknown_ages_do_not_enter_statistics(monkeypatch):
    rows = live_rows(monkeypatch, frame_stamps(3))
    rows[0]["timestamp_valid"] = False
    result = freshness(rows)
    assert result["full_observation_age_ms"][:2] == [None, None]
    assert result["unknown_full_observation_age_frames"] == 2
    assert result["full_observation_age"]["max_ms"] == 300.0
    rows[2]["timestamp_valid"] = False
    assert freshness(rows)["full_observation_age"] is None


def write_trial(directory, base, rows, policy):
    directory.mkdir()
    with (directory / "frames.jsonl").open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    indices = [0, len(rows) - 1]
    for index in indices:
        Image.fromarray(np.full((240, 320, 3), 90, np.uint8)).save(
            directory / f"{index:04d}-raw.png"
        )
    report = dict(
        policy=policy,
        machine="armv6l",
        opencv="4.x",
        model="m",
        model_sha256=sha256(base / "models/m.tflite"),
        bundle_sha256=sha256(base / "bundle.json"),
        script_sha256=sha256(ROOT / "scripts/pi_balloon_live_schedule_bench.py"),
        live_helper_sha256=sha256(ROOT / "scripts/pi_balloon_live_bench.py"),
        scheduler_sha256=sha256(base / "dtr/research_search_schedule.py"),
        region_library_sha256="region",
        runtime_library_sha256="runtime",
        interval_ns=SearchSchedule.interval_ns,
        max_bright_frames=SearchSchedule.max_bright_frames,
        camera_rotation_degrees=180,
        requested_camera_fps=10,
        requested_sensor_size=[2592, 1944],
        processing_size=[320, 240],
        queue=False,
        saved_sample_indices=indices,
        **trial_summary(rows, 2.5),
        files={p.name: sha256(p) for p in directory.iterdir()},
        camera_used=True,
        accuracy_measured=False,
        temporal_recall_measured=False,
        hard_reacquisition_deadline_proven=False,
        test_evaluated=False,
        deployment_approved=False,
        flight_commands=None,
    )
    write_json(directory / "report.json", report)
    return report


@pytest.fixture
def trials(tmp_path, monkeypatch):
    base, results = tmp_path / "base", tmp_path / "results"
    (base / "dtr").mkdir(parents=True)
    (base / "models").mkdir()
    (base / "models/m.tflite").write_bytes(b"model")
    shutil.copy(ROOT / "src/dtr/research_search_schedule.py", base / "dtr")
    write_json(base / "bundle.json", dict(files={}))
    write_json(base / "inputs.json", dict(models=dict(m=dict(path="models/m.tflite"))))
    monkeypatch.setattr(reviewer, "verify_bundle", lambda root: None)
    results.mkdir()
    for name, policy in (
        ("a-full", "full"),
        ("b-periodic", "periodic"),
        ("c-periodic", "periodic"),
        ("d-full", "full"),
    ):
        rows = live_rows(monkeypatch, frame_stamps(5), policy)
        write_trial(results / name, base, rows, policy)
    return base, results


def test_reviewer_recomputes_live_schedule_and_freshness(trials):
    result = reviewer.review(*trials)
    periodic, full = result["pooled"]["periodic"], result["pooled"]["full"]
    assert periodic["mode_counts"] == dict(mser_direct=6, mser_bright=4)
    assert full["mode_counts"] == dict(mser_direct=10, mser_bright=0)
    assert periodic["full_observation_age"]["max_ms"] == 550.0
    assert full["full_observation_age"]["max_ms"] == 300.0
    assert result["live_fps_ratio"] == 1.0
    assert result["flight_qualified"] is False


@pytest.mark.parametrize(
    "field,value",
    [("full_observation_age_ms", [300.0] * 5), ("mode_counts", {}), ("observed_live_fps", 9.0)],
)
def test_reviewer_rejects_summary_not_matching_raw_rows(trials, field, value):
    base, results = trials
    path = results / "b-periodic/report.json"
    report = json.loads(path.read_text())
    report[field] = value
    write_json(path, report)
    with pytest.raises(ValueError):
        reviewer.review(base, results)


def test_reviewer_rejects_tampered_frame_log(trials):
    base, results = trials
    log = results / "c-periodic/frames.jsonl"
    rows = [json.loads(s) for s in log.read_text().splitlines()]
    rows[1]["mode"] = "mser_direct"
    log.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="hash"):
        reviewer.review(base, results)


def test_reviewer_requires_two_trials_per_policy(trials):
    base, results = trials
    shutil.rmtree(results / "d-full")
    with pytest.raises(ValueError, match="two trials"):
        reviewer.review(base, results)
