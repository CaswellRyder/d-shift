"""Replay the other team's RGB-distance pixel detector on balloon frames; no camera or actuation.

The supplied module is imported by path and only its ``Detector`` is used, never its
camera/server ``main``. Its file is hash-checked before and after the run. Target colors
are supplied by the caller (they ship with a yellow-goal color only).
"""
import argparse
import hashlib
import importlib.util
import json
import platform
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

LABELS = ("red_balloon", "blue_balloon")
# demo.py's MIN_SHAPE_SIZE is 200 px at its 540x540 working size; keep the same image fraction.
NATIVE_AREA, NATIVE_MIN = 540 * 540, 200


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_pixel(path):
    spec = importlib.util.spec_from_file_location("pixel_candidate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def color(text):
    values = tuple(int(v) for v in text.split(","))
    if len(values) != 3 or not all(0 <= v <= 255 for v in values):
        raise argparse.ArgumentTypeError("Expected R,G,B in 0..255")
    return values


def min_size(width, height):
    return max(1, round(NATIVE_MIN * width * height / NATIVE_AREA))


def boxes_to_detections(boxes, label):
    """Inclusive pixel bounds -> half-open boxes; area stands in for the missing confidence."""
    rows = []
    for left, top, right, bottom in (tuple(int(v) for v in b) for b in boxes):
        box = [left, top, right + 1, bottom + 1]
        rows.append(dict(label=label, box=box, score=float((box[2] - box[0]) * (box[3] - box[1])),
                         accepted=True))
    return rows


def run_pass(detectors, images):
    rows = []
    for image in images:
        found, per_color = [], {}
        start = time.perf_counter()
        for label, detector in detectors.items():
            t = time.perf_counter()
            boxes, _ = detector.detect(image)
            per_color[label] = (time.perf_counter() - t) * 1000
            found.extend(boxes_to_detections(boxes, label))
        rows.append(dict(processing_ms=(time.perf_counter() - start) * 1000,
                         color_ms=per_color, detections=found))
    return rows


def statistics(values):
    return dict(mean_ms=float(np.mean(values)), p50_ms=float(np.median(values)),
                p95_ms=float(np.percentile(values, 95)), max_ms=float(np.max(values)),
                fps=1000 / float(np.mean(values)))


def health():
    result = {}
    for name, path in (("temperature_millicelsius", "/sys/class/thermal/thermal_zone0/temp"),
                       ("load_average", "/proc/loadavg")):
        if Path(path).exists():
            result[name] = Path(path).read_text().strip()
    try:
        result["throttled"] = subprocess.check_output(["vcgencmd", "get_throttled"], timeout=3,
                                                      text=True).strip()
    except (OSError, subprocess.SubprocessError):
        result["throttled"] = None
    return result


def bench(module, images, colors, threshold, repeats):
    height, width = images[0].shape[:2]
    size = min_size(width, height)
    detectors = {label: module.Detector(colors[label], threshold, size, "numpy") for label in LABELS}
    run_pass(detectors, images)  # warm-up; results discarded
    passes = [run_pass(detectors, images) for _ in range(repeats)]
    for rows in passes[1:]:
        if [r["detections"] for r in rows] != [r["detections"] for r in passes[0]]:
            raise RuntimeError("Pixel detector output changed between repeats")
    times = [r["processing_ms"] for rows in passes for r in rows]
    return dict(width=width, height=height, threshold=threshold, min_shape_size=size,
                repeats=repeats, timing=statistics(times),
                color_timing={label: statistics([r["color_ms"][label] for rows in passes for r in rows])
                              for label in LABELS},
                frames=[dict(detections=r["detections"],
                             processing_ms=[p[i]["processing_ms"] for p in passes])
                        for i, r in enumerate(passes[0])])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pixel", required=True, help="Path to the unmodified pixel demo.py")
    parser.add_argument("--pixel-sha256", required=True)
    parser.add_argument("--base", required=True, help="Bundle with inputs.json and frames/")
    parser.add_argument("--red", type=color, required=True)
    parser.add_argument("--blue", type=color, required=True)
    parser.add_argument("--threshold", type=float, action="append", required=True)
    parser.add_argument("--native-threshold", type=float, help="Also time this at 640x480 (no accuracy)")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 20:
        parser.error("--repeats must be 1..20")
    if sha256(args.pixel) != args.pixel_sha256:
        raise SystemExit("Pixel module hash mismatch")
    base = Path(args.base)
    inputs = json.loads((base / "inputs.json").read_text())
    images = [np.ascontiguousarray(np.asarray(Image.open(base / f["path"]).convert("RGB")))
              for f in inputs["frames"]]
    module = load_pixel(args.pixel)
    colors = dict(red_balloon=args.red, blue_balloon=args.blue)
    before = health()
    started = time.perf_counter()
    runs = [bench(module, images, colors, t, args.repeats) for t in args.threshold]
    native = None
    if args.native_threshold is not None:
        large = [cv2.resize(i, (640, 480), interpolation=cv2.INTER_LINEAR) for i in images]
        native = bench(module, large, colors, args.native_threshold, args.repeats)
        for frame in native["frames"]:
            del frame["detections"]  # upsampled pixels: timing only
    elapsed = time.perf_counter() - started
    if sha256(args.pixel) != args.pixel_sha256:
        raise SystemExit("Pixel module changed during the run")
    report = dict(
        purpose="Pixel-method balloon replay; development frames, NOT flight qualification",
        deployment_approved=False, flight_commands=None, test_evaluated=False,
        pixel=dict(path=str(args.pixel), sha256=args.pixel_sha256, backend="numpy",
                   have_numba=bool(module.HAVE_NUMBA)),
        colors={k: list(v) for k, v in colors.items()},
        inputs_sha256=sha256(base / "inputs.json"),
        frames=[dict(path=f["path"], sha256=sha256(base / f["path"])) for f in inputs["frames"]],
        runs=runs, native_640x480_timing=native, wall_seconds=elapsed,
        platform=dict(machine=platform.machine(), python=platform.python_version(),
                      numpy=np.__version__, opencv=cv2.__version__),
        health_before=before, health_after=health())
    Path(args.output).write_text(json.dumps(report, indent=1))
    for run in runs:
        print(f"thr {run['threshold']:g}: {run['timing']['mean_ms']:.2f} ms mean, "
              f"{run['timing']['fps']:.2f} FPS")
    if native:
        print(f"640x480 thr {native['threshold']:g}: {native['timing']['mean_ms']:.2f} ms mean")


if __name__ == "__main__":
    main()
