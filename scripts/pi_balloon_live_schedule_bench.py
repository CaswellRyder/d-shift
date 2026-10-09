"""Bounded live camera timing for periodic search; freshness, not recall or a deadline."""

import argparse
import json
import os
from pathlib import Path
import platform
import resource
import time

import cv2
import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, write_json
from dtr.native_balloon_regions import NativeBalloonRegions
from dtr.research_search_schedule import SearchSchedule
from dtr.runtime import Predictor

try:
    from pi_balloon_live_bench import run_camera, validate_camera_config
    from pi_balloon_schedule_bench import scheduled_frame
    from pi_balloon_search_bench import check_predictions, process_frame
    from pi_red_blue_bench import health, statistics, verify_bundle
except ModuleNotFoundError:
    from scripts.pi_balloon_live_bench import run_camera, validate_camera_config
    from scripts.pi_balloon_schedule_bench import scheduled_frame
    from scripts.pi_balloon_search_bench import check_predictions, process_frame
    from scripts.pi_red_blue_bench import health, statistics, verify_bundle


def boottime_ns():
    return time.clock_gettime_ns(time.CLOCK_BOOTTIME)


def full_observation_ages(rows):
    """Age at each result of the newest full-search frame's sensor exposure.

    A full search with an invalid sensor timestamp makes the age unknown until the
    next valid full search; an older full observation is never relabeled fresh.
    """
    latest, ages = None, []
    for row in rows:
        if row["mode"] == "mser_direct":
            latest = row["sensor_timestamp_ns"] if row["timestamp_valid"] else None
        ages.append(None if latest is None else (row["result_boottime_ns"] - latest) / 1e6)
    return ages


def freshness(rows):
    """Summaries over one trial's ordered rows; all clocks are CLOCK_BOOTTIME."""
    for row in rows:
        if not (
            row["capture_completed_boottime_ns"]
            <= row["start_ns"]
            <= row["end_ns"]
            <= row["result_boottime_ns"]
        ):
            raise ValueError("Scheduler timestamps fall outside the camera frame")
    full = [r for r in rows if r["mode"] == "mser_direct"]
    sensors = [r["sensor_timestamp_ns"] for r in full if r["timestamp_valid"]]
    ages = full_observation_ages(rows)
    known = [a for a in ages if a is not None]
    return dict(
        full_result_intervals_ms=[
            (b["result_boottime_ns"] - a["result_boottime_ns"]) / 1e6
            for a, b in zip(full, full[1:])
        ],
        full_sensor_intervals_ms=[(b - a) / 1e6 for a, b in zip(sensors, sensors[1:])],
        full_observation_age_ms=ages,
        unknown_full_observation_age_frames=len(ages) - len(known),
        full_observation_age=statistics(known) if known else None,
    )


def trial_summary(rows, elapsed):
    """Every report field derived from raw rows, so review recomputes the same values."""
    valid = [r["sensor_to_result_ms"] for r in rows if r["timestamp_valid"]]
    full_starts = [r["start_ns"] for r in rows if r["mode"] == "mser_direct"]
    return dict(
        frames=len(rows),
        elapsed_s=elapsed,
        observed_live_fps=len(rows) / elapsed,
        mode_counts={k: sum(r["mode"] == k for r in rows) for k in ("mser_direct", "mser_bright")},
        timing={
            key: statistics([r[key] for r in rows])
            for key in (
                "processing_ms",
                "scheduled_processing_ms",
                "search_ms",
                "inference_ms",
                "selection_ms",
                "capture_and_convert_ms",
                "frame_wall_ms",
            )
        },
        sensor_to_result=statistics(valid) if valid else None,
        invalid_timestamp_frames=sum(not r["timestamp_valid"] for r in rows),
        increasing_sensor_timestamps=all(r["sensor_timestamp_increasing"] for r in rows),
        sensor_clock="CLOCK_BOOTTIME",
        scheduler_clock="CLOCK_BOOTTIME",
        full_start_intervals_ms=[(b - a) / 1e6 for a, b in zip(full_starts, full_starts[1:])],
        max_dispatch_lateness_ms=max(r["due_lateness_ns"] for r in rows) / 1e6,
        **freshness(rows),
        mean_neural_calls=float(np.mean([r["neural_calls"] for r in rows])),
        accepted_counts={
            label: sum(d["accepted"] and d["label"] == label for r in rows for d in r["detections"])
            for label in ("red_balloon", "blue_balloon")
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "region-library", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--policy", choices=("full", "periodic"), required=True)
    parser.add_argument("--model", default="new-views-42")
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--fps", type=float, default=10)
    args = parser.parse_args()
    if not 2 <= args.count <= 120 or not 3 <= args.seconds <= 60 or not 1 <= args.fps <= 15:
        parser.error("Require count 2..120, seconds 3..60, fps 1..15")
    root, out = Path(args.base).resolve(), Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    verify_bundle(root)
    native = NativeBalloonRegions(args.region_library, direct=True)
    receipt = read_json(native.path.with_suffix(".json"))
    if receipt["source_sha256"] != sha256(root / "dtr/native_balloon_regions.c"):
        raise ValueError("Native reducer source mismatch")
    for key in ("DTR_RED_BLUE_COLOR_LOOKUP", "DTR_RED_BLUE_LOOKUP_LIBRARY"):
        os.environ.pop(key, None)
    cv2.setNumThreads(1)
    inputs = read_json(root / "inputs.json")
    golden = read_json(root / "search-golden.json")[args.model]
    model = root / inputs["models"][args.model]["path"]
    predictor = Predictor(model, allow_unvalidated=True)
    # Both policy goldens pass before the camera opens; same warmup for both policies.
    deltas = []
    for mode in ("mser_direct", "mser_bright"):
        expected = golden["mser" if mode == "mser_direct" else mode]
        if len(expected) != len(inputs["frames"]) or not expected:
            raise ValueError("Incomplete policy goldens")
        for frame, truth in zip(inputs["frames"], expected, strict=True):
            rgb = np.asarray(Image.open(root / frame["path"]).convert("RGB"))
            observed = process_frame(rgb, predictor, mode, native)
            deltas.append(check_predictions(observed["detections"], truth["detections"]))
    scheduler = SearchSchedule()

    def infer(rgb):
        return scheduled_frame(rgb, predictor, native, scheduler, args.policy, boottime_ns)

    from libcamera import Transform
    from picamera2 import Picamera2

    camera = Picamera2()
    before = health()
    try:
        camera.configure(
            camera.create_video_configuration(
                main={"size": (320, 240), "format": "RGB888"},
                sensor={"output_size": (2592, 1944), "bit_depth": 10},
                controls={"FrameRate": args.fps},
                buffer_count=2,
                queue=False,
                transform=Transform(hflip=True, vflip=True),
            )
        )
        config = camera.camera_configuration()
        configured_transform = str(config["transform"])
        validate_camera_config(config)
        camera.start()
        time.sleep(1)
        cpu_start = time.process_time()
        rows, samples, elapsed = run_camera(camera, infer, args.count, args.seconds, boottime_ns)
        cpu = time.process_time() - cpu_start
    finally:
        camera.close()
    after = health()
    summary = trial_summary(rows, elapsed)
    out.mkdir(parents=True, exist_ok=False)
    with (out / "frames.jsonl").open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    for index, rgb, detections in samples:
        Image.fromarray(rgb).save(out / f"{index:04d}-raw.png")
        annotated = rgb.copy()
        for d in detections:
            if d["accepted"]:
                x1, y1, x2, y2 = map(int, d["box"])
                cv2.rectangle(annotated, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 0), 1)
                cv2.putText(
                    annotated,
                    d["label"],
                    (x1, max(12, y1)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (0, 255, 0),
                    1,
                )
        Image.fromarray(annotated).save(out / f"{index:04d}-annotated.png")
    report = dict(
        scope="Unlabeled live scheduler timing and observation age; not recall or a deadline",
        policy=args.policy,
        machine=platform.machine(),
        opencv=cv2.__version__,
        model=args.model,
        model_sha256=sha256(model),
        bundle_sha256=sha256(root / "bundle.json"),
        script_sha256=sha256(__file__),
        live_helper_sha256=sha256(Path(__file__).with_name("pi_balloon_live_bench.py")),
        scheduler_sha256=sha256(root / "dtr/research_search_schedule.py"),
        region_library_sha256=sha256(native.path),
        region_build_receipt=receipt,
        runtime_library_sha256=sha256(os.environ["DTR_TFLITE_LIBRARY"])
        if os.environ.get("DTR_TFLITE_LIBRARY")
        else None,
        interval_ns=SearchSchedule.interval_ns,
        max_bright_frames=SearchSchedule.max_bright_frames,
        camera_rotation_degrees=180,
        configured_transform=configured_transform,
        camera_config=str(config),
        requested_camera_fps=args.fps,
        requested_sensor_size=[2592, 1944],
        processing_size=[320, 240],
        queue=False,
        max_pre_camera_golden_score_delta=max(deltas),
        requested_frames=args.count,
        saved_sample_indices=[index for index, _, _ in samples],
        sample_policy="First/last plus first eight accepted-detection frames; saved after timing",
        timed_loop_cpu_seconds=cpu,
        stopped_at_time_limit=len(rows) < args.count,
        **summary,
        health_before=before,
        health_after=after,
        max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        files={p.name: sha256(p) for p in out.iterdir() if p.is_file()},
        camera_used=True,
        accuracy_measured=False,
        temporal_recall_measured=False,
        hard_reacquisition_deadline_proven=False,
        test_evaluated=False,
        deployment_approved=False,
        flight_commands=None,
    )
    write_json(out / "report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "full_observation_age_ms"}, indent=2))


if __name__ == "__main__":
    main()
