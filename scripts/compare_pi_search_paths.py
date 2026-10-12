"""Bounded, isolated search-path experiments; never a flight qualification.

Uses frozen bundle modules/weights and fixed historical pixel settings. Generate
host goldens first, then replay on the Pi; live runs verify those goldens before
opening the camera. Outputs are new directories, never overwrites.
"""

import argparse
import importlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

METHODS = ("mser", "pixel1", "pixel2")
# Live only: stride-2 pixel search between full searches, dispatched by the
# bundle's frozen SearchSchedule (500 ms or two fast frames).
LIVE_METHODS = METHODS + ("recovery",)


def summarize(rows, elapsed=None):
    if not rows:
        raise ValueError("Empty timing sample")
    result = {}
    for key in ("processing_ms", "search_ms", "inference_ms", "sensor_to_result_ms",
                "capture_and_convert_ms", "frame_wall_ms"):
        values = [row[key] for row in rows if row.get(key) is not None]
        if values:
            if not np.isfinite(values).all() or min(values) < 0:
                raise ValueError("Invalid timing sample")
            result[key] = dict(mean=float(np.mean(values)), p95=float(np.percentile(values, 95)),
                               maximum=float(max(values)))
    mean = result["processing_ms"]["mean"]
    if mean <= 0:
        raise ValueError("Non-positive processing duration")
    result["processing_fps"] = 1000 / mean
    result["neural_calls"] = float(np.mean([r["neural_calls"] for r in rows]))
    methods = [r["search_method"] for r in rows if "search_method" in r]
    if methods:
        result["full_search_fraction"] = methods.count("mser") / len(methods)
        done = [r["sensor_timestamp_ns"] / 1e6 + r["sensor_to_result_ms"] for r in rows
                if r.get("search_method") == "mser" and r.get("timestamp_valid")]
        result["max_full_search_gap_ms"] = float(max(np.diff(done))) if len(done) > 1 else None
    if elapsed is not None:
        if not np.isfinite(elapsed) or elapsed <= 0:
            raise ValueError("Invalid wall duration")
        result["live_fps"] = len(rows) / elapsed
        result["valid_timestamps"] = sum(r["timestamp_valid"] for r in rows)
        result["all_sensor_timestamps_increasing"] = all(
            r["sensor_timestamp_increasing"] for r in rows)
    return result


def recovery_search(infer, clock_ns=time.monotonic_ns):
    """Run pixel2 between full searches; reason and method are logged per row."""
    from dtr.research_search_schedule import SearchSchedule

    schedule = SearchSchedule()

    def run(rgb):
        decision = schedule.begin(clock_ns())
        method = "pixel2" if decision["mode"] == "mser_bright" else "mser"
        try:
            observed = infer(rgb, method)
        except Exception:
            schedule.finish(clock_ns(), success=False)
            raise
        schedule.finish(clock_ns())
        return dict(observed, search_method=method, search_reason=decision["reason"])

    return run


def validate_identity(actual, expected):
    if actual != expected:
        raise ValueError("Golden model, inputs, bundle, or source identity mismatch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--region-library", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("golden", "replay", "live"), required=True)
    parser.add_argument("--golden")
    parser.add_argument("--method", choices=LIVE_METHODS, default="mser")
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--seconds", type=float, default=30)
    args = parser.parse_args()
    if args.mode != "golden" and not args.golden:
        parser.error("--golden required for hardware experiments")
    if args.method == "recovery" and args.mode != "live":
        parser.error("recovery is a live-camera method")
    if not 2 <= args.count <= 120 or not 3 <= args.seconds <= 60:
        parser.error("Bound camera work to 2..120 frames and 3..60 seconds")
    root, out = Path(args.base).resolve(), Path(args.output).resolve()
    if out.exists():
        raise FileExistsError(out)
    # Prefer frozen bundle implementation over the concurrent checkout.
    sys.path.insert(0, str(root))
    import cv2
    from PIL import Image
    from dtr.data import read_json, sha256, write_json
    from dtr.native_balloon_regions import NativeBalloonRegions
    from dtr.runtime import Predictor
    from pi_red_blue_bench import verify_bundle, health
    from pi_balloon_search_bench import process_frame, check_predictions
    from pi_balloon_live_bench import run_camera, validate_camera_config
    from research_pixel_proposals import CvPixelMasks, process_frame as pixel_frame

    verify_bundle(root)
    cv2.setNumThreads(1)
    for key in ("DTR_RED_BLUE_COLOR_LOOKUP", "DTR_RED_BLUE_LOOKUP_LIBRARY"):
        os.environ.pop(key, None)
    inputs = read_json(root / "inputs.json")
    alias = "soft-admit-47"
    model = root / inputs["models"][alias]["path"]
    native = NativeBalloonRegions(args.region_library, direct=True)
    if read_json(native.path.with_suffix(".json"))["source_sha256"] != sha256(
            root / "dtr/native_balloon_regions.c"):
        raise ValueError("Native source mismatch")
    predictor = Predictor(model, allow_unvalidated=True)
    masks = CvPixelMasks(((178, 28, 48), (26, 86, 179)), 75)
    sources = {name: sha256(importlib.import_module(name).__file__) for name in (
        "research_pixel_proposals", "research_balloon_search", "research_balloon_search_fast",
        "pi_balloon_search_bench", "pi_balloon_live_bench", "dtr.runtime",
        "dtr.native_balloon_regions", "dtr.vision", "dtr.research_search_schedule")}
    identity = dict(model=alias, model_sha256=sha256(model), inputs_sha256=sha256(root / "inputs.json"),
                    bundle_sha256=sha256(root / "bundle.json"), sources=sources,
                    script_sha256=sha256(__file__), threshold=75, min_area=53,
                    colors=[[178, 28, 48], [26, 86, 179]])

    def infer(rgb, method):
        if method == "mser":
            return process_frame(rgb, predictor, "mser_direct", native)
        return pixel_frame(rgb, predictor, masks, 53, stride=1 if method == "pixel1" else 2)

    frames = inputs["frames"]
    images = [np.asarray(Image.open(root / f["path"]).convert("RGB")) for f in frames]
    original = read_json(root / "search-golden.json")[alias]["mser"]
    if not images or len(images) != len(original):
        raise ValueError("Incomplete input/golden coverage")
    out.mkdir(parents=True)
    if args.mode == "golden":
        goldens = {}
        for method in METHODS:
            goldens[method] = [infer(rgb, method)["detections"] for rgb in images]
            if method == "mser":
                for actual, reference in zip(goldens[method], original, strict=True):
                    check_predictions(actual, reference["detections"])
        write_json(out / "golden.json", dict(identity=identity, predictions=goldens))
        print(json.dumps(dict(golden=str(out / "golden.json"), frames=len(frames))), flush=True)
        return
    expected = read_json(args.golden)
    validate_identity(identity, expected["identity"])
    sequence = ("mser", "pixel1", "pixel2", "pixel2", "pixel1", "mser") if args.mode == "replay" else (args.method,)
    results = []
    for trial, method in enumerate(sequence):
        components = ("mser", "pixel2") if method == "recovery" else (method,)
        max_delta = 0.0
        before = health()
        for component in components:
            for rgb, prediction in zip(images, expected["predictions"][component], strict=True):
                max_delta = max(max_delta, check_predictions(
                    infer(rgb, component)["detections"], prediction))
        elapsed = None
        if args.mode == "replay":
            rows = [dict(index=i, **infer(rgb, method)) for i, rgb in enumerate(images)]
            for row, prediction in zip(rows, expected["predictions"][method], strict=True):
                max_delta = max(max_delta, check_predictions(row["detections"], prediction))
        else:
            from picamera2 import Picamera2
            from libcamera import Transform
            camera = Picamera2()
            try:
                camera.configure(camera.create_video_configuration(
                    main={"size": (320, 240), "format": "RGB888"},
                    sensor={"output_size": (2592, 1944), "bit_depth": 10},
                    controls={"FrameRate": 10}, buffer_count=2, queue=False,
                    transform=Transform(hflip=True, vflip=True)))
                validate_camera_config(camera.camera_configuration())
                camera.start()
                time.sleep(1)
                run = (recovery_search(infer) if method == "recovery"
                       else lambda rgb: infer(rgb, method))
                rows, samples, elapsed = run_camera(camera, run, args.count, args.seconds)
            finally:
                camera.close()
            for index, rgb, _ in samples:
                Image.fromarray(rgb).save(out / f"sample-{index:03d}.png")
        path = out / f"{trial}-{method}.json"
        write_json(path, dict(rows=rows))
        report = dict(trial=trial, method=method, frames=len(rows), summary=summarize(rows, elapsed),
                      elapsed_seconds=elapsed, max_host_pi_score_delta=max_delta,
                      health_before=before, health_after=health(), rows_sha256=sha256(path))
        results.append(report)
        print(json.dumps(report), flush=True)
    write_json(out / "report.json", dict(identity=identity, trials=results,
               machine=platform.machine(), opencv=cv2.__version__, mode=args.mode,
               runtime_library_sha256=sha256(os.environ["DTR_TFLITE_LIBRARY"])
               if os.environ.get("DTR_TFLITE_LIBRARY") else None,
               region_library_sha256=sha256(native.path), golden_sha256=sha256(args.golden),
               camera_used=args.mode == "live", camera_sensor=[2592, 1944] if elapsed else None,
               camera_output=[320, 240] if elapsed else None, requested_camera_fps=10 if elapsed else None,
               camera_queue=False if elapsed else None, rotation_degrees=180 if elapsed else None,
               preview_enabled=False, deployment_approved=False, test_evaluated=False,
               flight_commands=None, scope="Development replay or unlabeled live timing; not qualification"))


if __name__ == "__main__":
    main()
