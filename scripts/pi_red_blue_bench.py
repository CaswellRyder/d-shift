"""One model and policy per process: crop/replay/live timing, no actuation."""
import argparse
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import time

import cv2
import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, write_json
from dtr.runtime import Predictor
from dtr.temporal import TemporalVision
from dtr.vision import RED_BLUE_PROFILE, observe, validate_model_profile


def verify_bundle(root):
    root = Path(root).resolve()
    receipt = read_json(root / "bundle.json")
    for relative, digest in receipt["files"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or sha256(path) != digest:
            raise ValueError("Bundle file changed or escaped root")
    return receipt


def statistics(values):
    if not values or not np.isfinite(values).all() or min(values) < 0:
        raise ValueError("Require finite nonnegative timings")
    return dict(mean_ms=float(np.mean(values)), p50_ms=float(np.median(values)),
                p95_ms=float(np.percentile(values, 95)), max_ms=float(max(values)))


def health():
    result = {}
    for name, path in (("temperature_millicelsius", "/sys/class/thermal/thermal_zone0/temp"),
                       ("load_average", "/proc/loadavg")):
        if Path(path).exists():
            result[name] = Path(path).read_text().strip()
    try:
        result["throttled"] = subprocess.check_output(["vcgencmd", "get_throttled"], timeout=3, text=True).strip()
    except (OSError, subprocess.SubprocessError):
        result["throttled"] = None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--model", required=True, help="Alias in inputs.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("crops", "replay", "camera"), default="crops")
    parser.add_argument("--policy", choices=("stateless", "temporal"), default="stateless")
    parser.add_argument("--count", type=int, default=80)
    parser.add_argument("--budget", type=int, default=12)
    parser.add_argument("--fps", type=float, default=10)
    args = parser.parse_args()
    if not 1 <= args.count <= 600 or not 1 <= args.budget <= 12 or not 1 <= args.fps <= 30:
        parser.error("Require count 1..600, budget 1..12, fps 1..30")
    if args.mode != "camera" and args.policy != "stateless":
        parser.error("Temporal timing requires real camera timestamps, not repeated stills")
    root, output = Path(args.base).resolve(), Path(args.output)
    verify_bundle(root)
    inputs = read_json(root / "inputs.json")
    model_record = inputs["models"][args.model]
    if output.exists() or output.with_suffix(".frames.jsonl").exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    predictor = Predictor(root / model_record["path"], allow_unvalidated=True)
    validate_model_profile(predictor.metadata, RED_BLUE_PROFILE)
    crops = [np.asarray(Image.open(root / r["path"]).convert("RGB")) for r in inputs["crops"]]
    parity = []
    for rgb, expected in zip(crops, model_record["golden"]):
        prediction = predictor.predict(rgb)
        delta = float(np.max(np.abs(np.asarray(prediction["scores"])-expected["scores"])))
        parity.append(delta)
        if (prediction["label"] != expected["label"] or prediction["accepted"] != expected["accepted"]
                or delta > model_record["score_tolerance"]):
            raise ValueError("Host/Pi prediction parity failed")
    report = dict(host=platform.node(), machine=platform.machine(), mode=args.mode, policy=args.policy,
                  model=args.model, model_sha256=sha256(root / model_record["path"]),
                  bundle_sha256=sha256(root / "bundle.json"), input_sha256=sha256(root / "inputs.json"),
                  runtime=type(predictor.interpreter).__module__,
                  runtime_version=getattr(predictor.interpreter, "version", None),
                  max_golden_score_delta=max(parity), budget=args.budget, health_before=health(),
                  deployment_approved=False, test_evaluated=False, flight_commands=None)
    library = os.environ.get("DTR_TFLITE_LIBRARY")
    report["runtime_library_sha256"] = sha256(library) if library else None
    records = []
    if args.mode == "crops":
        for index in range(args.count):
            prediction = predictor.predict(crops[index % len(crops)])
            records.append(dict(index=index, processing_ms=prediction["latency_ms"]))
        report["scope"] = "Warmed resize + inference + softmax; excludes camera and frame search"
    elif args.mode == "replay":
        frames = [np.asarray(Image.open(root / r["path"]).convert("RGB")) for r in inputs["frames"]]
        for rgb in frames:
            observe(rgb, predictor, limit=args.budget, profile=RED_BLUE_PROFILE)
        for index in range(args.count):
            position = index % len(frames)
            result = observe(frames[position], predictor, limit=args.budget, profile=RED_BLUE_PROFILE)
            records.append(dict(index=index, source=inputs["frames"][position]["source"], **result))
        report["scope"] = "Repeated off-domain development stills, processing only; not live FPS or held-out accuracy"
    else:
        from libcamera import Transform
        from picamera2 import Picamera2
        tracker = TemporalVision(predictor, budget=args.budget, proposal_profile=RED_BLUE_PROFILE)
        camera = Picamera2()
        camera.configure(camera.create_video_configuration(main=dict(size=(320, 240), format="RGB888"),
                         controls={"FrameRate": args.fps}, buffer_count=2,
                         transform=Transform(hflip=True, vflip=True)))
        report.update(camera_rotation_degrees=180, requested_camera_fps=args.fps,
                      camera_size=[320, 240], camera_config=str(camera.camera_configuration()))
        camera.start()
        try:
            time.sleep(1)
            start = time.monotonic()
            for index in range(args.count):
                frame_start = time.perf_counter()
                request = camera.capture_request()
                try:
                    rgb = cv2.cvtColor(request.make_array("main"), cv2.COLOR_BGR2RGB)
                    sensor = request.get_metadata().get("SensorTimestamp")
                finally:
                    request.release()
                processing_start = time.perf_counter()
                result = (tracker.observe(rgb) if args.policy == "temporal" else
                          observe(rgb, predictor, limit=args.budget, profile=RED_BLUE_PROFILE))
                # libcamera SensorTimestamp uses Linux CLOCK_BOOTTIME, not
                # CLOCK_MONOTONIC (the two diverge after system suspend).
                completed = time.clock_gettime_ns(time.CLOCK_BOOTTIME)
                age = (completed-sensor)/1e6 if sensor is not None else None
                records.append(dict(index=index, **result, sensor_timestamp_ns=sensor,
                                    capture_and_convert_ms=(processing_start-frame_start)*1000,
                                    output_age_ms=age if age is not None and age >= 0 else None))
                if index in (0, args.count-1):
                    Image.fromarray(rgb).save(output.with_name(f"{output.stem}-{index:04}.png"))
            report.update(wall_seconds=time.monotonic()-start)
            report["observed_loop_fps"] = args.count/report["wall_seconds"]
        finally:
            camera.stop()
            camera.close()
        stamps = [r["sensor_timestamp_ns"] for r in records]
        report.update(scope="Unlabeled live camera timing; NO detection accuracy or flight qualification",
                      increasing_sensor_timestamps=all(a is not None and b is not None and b > a
                                                      for a, b in zip(stamps, stamps[1:])),
                      output_age_clock="libcamera SensorTimestamp vs Linux CLOCK_BOOTTIME; invalid ages omitted")
        ages = [r["output_age_ms"] for r in records if r["output_age_ms"] is not None]
        report["output_age"] = statistics(ages) if ages else None
        report["missing_or_invalid_ages"] = len(records)-len(ages)
    report.update(timing=statistics([r["processing_ms"] for r in records]), frames=len(records),
                  health_after=health(), max_rss_platform_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  rss_units="KiB on Linux, bytes on macOS")
    with output.with_suffix(".frames.jsonl").open("x") as stream:
        for record in records:
            stream.write(json.dumps(record, allow_nan=False)+"\n")
    write_json(output, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
