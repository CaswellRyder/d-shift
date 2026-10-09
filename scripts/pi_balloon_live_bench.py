"""Bounded current-student live timing, without flight commands or accuracy claims."""

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
from dtr.runtime import Predictor

try:
    from pi_balloon_search_bench import check_predictions, process_frame
    from pi_red_blue_bench import health, statistics, verify_bundle
except ModuleNotFoundError:
    from scripts.pi_balloon_search_bench import check_predictions, process_frame
    from scripts.pi_red_blue_bench import health, statistics, verify_bundle


def capture_rgb(camera, timeout=3.0):
    """Own RGB pixels before releasing the request; RGB888 is BGR in memory."""
    request = camera.wait(camera.capture_request(wait=False), timeout=timeout)
    try:
        array = request.make_array("main")
        if array.dtype != np.uint8 or array.shape != (240, 320, 3):
            raise ValueError("Unexpected camera buffer format or size")
        rgb = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        metadata = dict(request.get_metadata())
    finally:
        request.release()
    return rgb, metadata


def validate_camera_config(config):
    transform = config["transform"]
    if (
        not transform.hflip
        or not transform.vflip
        or transform.transpose
        or tuple(config["main"]["size"]) != (320, 240)
        or config["main"]["format"] != "RGB888"
        or tuple(config["sensor"]["output_size"]) != (2592, 1944)
        or config["sensor"]["bit_depth"] != 10
        or config["queue"] is not False
    ):
        raise ValueError("Camera mode, format, queue or 180-degree transform changed")


def frame_timing(metadata, started_ns, acquired_ns, completed_ns, previous_sensor=None):
    if not 0 < started_ns <= acquired_ns <= completed_ns:
        raise ValueError("Invalid host boottime ordering")
    sensor = metadata.get("SensorTimestamp")
    valid = type(sensor) is int and 0 < sensor <= acquired_ns
    increasing = valid and (previous_sensor is None or sensor > previous_sensor)
    return dict(
        sensor_timestamp_ns=sensor,
        capture_started_boottime_ns=started_ns,
        capture_completed_boottime_ns=acquired_ns,
        result_boottime_ns=completed_ns,
        timestamp_valid=valid,
        sensor_timestamp_increasing=increasing,
        sensor_to_acquire_ms=(acquired_ns - sensor) / 1e6 if valid else None,
        sensor_to_result_ms=(completed_ns - sensor) / 1e6 if valid else None,
        exposure_us=metadata.get("ExposureTime"),
        analogue_gain=metadata.get("AnalogueGain"),
        frame_duration_us=metadata.get("FrameDuration"),
    )


def run_camera(camera, infer, count, seconds, clock_ns=None):
    """Keep samples/logs in memory; no disk I/O in the timed loop."""
    if not 2 <= count <= 120 or not 3 <= seconds <= 60:
        raise ValueError("Require 2..120 frames and 3..60 seconds")
    clock_ns = clock_ns or (lambda: time.clock_gettime_ns(time.CLOCK_BOOTTIME))
    rows, samples = [], []
    started = time.perf_counter()
    previous = None
    last_rgb = None
    for index in range(count):
        if time.perf_counter() - started >= seconds:
            break
        frame_start = time.perf_counter()
        start_ns = clock_ns()
        rgb, metadata = capture_rgb(camera)
        acquired_ns = clock_ns()
        acquired = time.perf_counter()
        observed = infer(rgb)
        completed_ns = clock_ns()
        done = time.perf_counter()
        timing = frame_timing(metadata, start_ns, acquired_ns, completed_ns, previous)
        if timing["timestamp_valid"]:
            previous = timing["sensor_timestamp_ns"]
        rows.append(
            dict(
                index=index,
                **observed,
                **timing,
                capture_and_convert_ms=(acquired - frame_start) * 1000,
                frame_wall_ms=(done - frame_start) * 1000,
            )
        )
        if index == 0:
            samples.append((index, rgb, observed["detections"]))
        last_rgb = rgb
    elapsed = time.perf_counter() - started
    if len(rows) < 2:
        raise ValueError("Too few live frames for timing")
    samples.append((rows[-1]["index"], last_rgb, rows[-1]["detections"]))
    return rows, samples, elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="new-views-42")
    parser.add_argument("--search", choices=("mser", "mser_fast"), default="mser_fast")
    parser.add_argument("--region-library")
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--fps", type=float, default=10)
    args = parser.parse_args()
    if not 2 <= args.count <= 120 or not 3 <= args.seconds <= 60 or not 1 <= args.fps <= 15:
        parser.error("Require count 2..120, seconds 3..60, fps 1..15")
    root, out = Path(args.base).resolve(), Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    verify_bundle(root)
    native = None
    if args.search == "mser_fast":
        if not args.region_library:
            parser.error("Fast search requires --region-library")
        native = NativeBalloonRegions(args.region_library)
        receipt = read_json(native.path.with_suffix(".json"))
        if receipt["source_sha256"] != sha256(root / "dtr/native_balloon_regions.c"):
            raise ValueError("Native region source receipt mismatch")
    elif args.region_library:
        parser.error("Region library is only used by mser_fast")
    for key in ("DTR_RED_BLUE_COLOR_LOOKUP", "DTR_RED_BLUE_LOOKUP_LIBRARY"):
        os.environ.pop(key, None)
    cv2.setNumThreads(1)
    inputs = read_json(root / "inputs.json")
    model = root / inputs["models"][args.model]["path"]
    predictor = Predictor(model, allow_unvalidated=True)

    def infer(rgb):
        return process_frame(rgb, predictor, args.search, native)

    golden = read_json(root / "search-golden.json")[args.model]["mser"]
    deltas = []
    for frame, expected in zip(inputs["frames"], golden):
        rgb = np.asarray(Image.open(root / frame["path"]).convert("RGB"))
        deltas.append(check_predictions(infer(rgb)["detections"], expected["detections"]))
    if not deltas or len(deltas) != len(inputs["frames"]) or len(golden) != len(deltas):
        raise ValueError("Incomplete pre-camera parity check")
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
        rows, samples, elapsed = run_camera(camera, infer, args.count, args.seconds)
    finally:
        camera.close()
    after = health()
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
        scope="Unlabeled live stateless timing; no comparative accuracy or flight qualification",
        machine=platform.machine(),
        opencv=cv2.__version__,
        model=args.model,
        search=args.search,
        model_sha256=sha256(model),
        bundle_sha256=sha256(root / "bundle.json"),
        script_sha256=sha256(__file__),
        region_library_sha256=sha256(native.path) if native else None,
        runtime_library_sha256=sha256(os.environ["DTR_TFLITE_LIBRARY"])
        if os.environ.get("DTR_TFLITE_LIBRARY")
        else None,
        camera_rotation_degrees=180,
        configured_transform=configured_transform,
        camera_config=str(config),
        requested_camera_fps=args.fps,
        requested_sensor_size=[2592, 1944],
        processing_size=[320, 240],
        queue=False,
        max_pre_camera_golden_score_delta=max(deltas),
        requested_frames=args.count,
        frames=len(rows),
        elapsed_s=elapsed,
        observed_live_fps=len(rows) / elapsed,
        stopped_at_time_limit=len(rows) < args.count,
        timing={
            key: statistics([r[key] for r in rows])
            for key in (
                "processing_ms",
                "search_ms",
                "inference_ms",
                "selection_ms",
                "capture_and_convert_ms",
                "frame_wall_ms",
            )
        },
        sensor_to_result=statistics(
            [r["sensor_to_result_ms"] for r in rows if r["timestamp_valid"]]
        )
        if any(r["timestamp_valid"] for r in rows)
        else None,
        invalid_timestamp_frames=sum(not r["timestamp_valid"] for r in rows),
        increasing_sensor_timestamps=all(r["sensor_timestamp_increasing"] for r in rows),
        sensor_clock="CLOCK_BOOTTIME",
        mean_neural_calls=float(np.mean([r["neural_calls"] for r in rows])),
        accepted_counts={
            label: sum(d["accepted"] and d["label"] == label for r in rows for d in r["detections"])
            for label in ("red_balloon", "blue_balloon")
        },
        health_before=before,
        health_after=after,
        max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        files={p.name: sha256(p) for p in out.iterdir() if p.is_file()},
        camera_used=True,
        accuracy_measured=False,
        test_evaluated=False,
        deployment_approved=False,
        flight_commands=None,
    )
    write_json(out / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
