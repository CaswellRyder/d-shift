"""Live Pi frame-rate probe: camera modes, per-stage cost, alternating schedules, zoom delay.

Timing only. No flight commands, no accuracy claims, no frames written to disk.
"""

import argparse
import json
import os
from pathlib import Path
import platform
import time

import cv2
import numpy as np

from dtr.goal_square import square_candidates
from dtr.runtime import Predictor
from dtr.vision import GREEN_PURPLE_HSV, GREEN_PURPLE_PROFILE, observe

FULL = (2592, 1944)
SENSOR_MODES = {"full": ((2592, 1944), 15), "binned": ((1296, 972), 40), "vga": ((640, 480), 58)}
WINDOW = 48  # Track-check window side, in 320x240 pixels.


def open_camera(sensor, fps):
    from libcamera import Transform
    from picamera2 import Picamera2

    camera = Picamera2()
    try:
        camera.configure(camera.create_video_configuration(
            main={"size": (320, 240), "format": "RGB888"},
            sensor={"output_size": sensor, "bit_depth": 10},
            controls={"FrameRate": fps},
            buffer_count=3,
            queue=False,
            transform=Transform(hflip=True, vflip=True),
        ))
        camera.start()
    except BaseException:
        camera.close()
        raise
    time.sleep(1.5)
    return camera


def grab(camera):
    request = camera.capture_request()
    try:
        array = request.make_array("main")
        if array.shape != (240, 320, 3):
            raise ValueError("Unexpected camera buffer size")
        rgb = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        metadata = dict(request.get_metadata())
    finally:
        request.release()
    return rgb, metadata


def summary(values):
    values = np.asarray(values, float)
    if not len(values):
        return None
    return dict(n=int(len(values)), mean=round(float(values.mean()), 2),
                p50=round(float(np.percentile(values, 50)), 2),
                p90=round(float(np.percentile(values, 90)), 2), max=round(float(values.max()), 2))


def rate(camera, seconds, work=None):
    """Frames per second delivered while running `work` on each frame (None = camera only)."""
    stamps, grab_ms, work_ms, sensor = [], [], [], []
    end = time.perf_counter() + seconds
    index = 0
    while time.perf_counter() < end:
        t0 = time.perf_counter()
        rgb, metadata = grab(camera)
        t1 = time.perf_counter()
        if work is not None:
            work(rgb, index)
        t2 = time.perf_counter()
        stamps.append(t2)
        grab_ms.append((t1 - t0) * 1000)
        work_ms.append((t2 - t1) * 1000)
        sensor.append(metadata.get("SensorTimestamp"))
        index += 1
    span = stamps[-1] - stamps[0]
    gaps = np.diff([s for s in sensor if s]) / 1e6
    return dict(frames=len(stamps), loop_fps=round((len(stamps) - 1) / span, 2),
                grab_ms=summary(grab_ms), work_ms=summary(work_ms),
                sensor_gap_ms=summary(gaps),
                exposure_us=metadata.get("ExposureTime"), gain=metadata.get("AnalogueGain"),
                scaler_crop=list(metadata.get("ScalerCrop", ())))


def track_check(rgb, center):
    """Cheap 'is it still there' test: green/purple pixels in a small window."""
    x, y = center
    half = WINDOW // 2
    window = rgb[max(0, y - half):y + half, max(0, x - half):x + half]
    hsv = cv2.cvtColor(window, cv2.COLOR_RGB2HSV)
    hits = [int(cv2.countNonZero(cv2.inRange(hsv, lo, hi))) for lo, hi in GREEN_PURPLE_HSV]
    return hits


def stage_costs(camera, predictor, seconds):
    rows = dict(balloon_full=[], balloon_search=[], balloon_classify=[], crops=[],
                goal_squares=[], hsv_full_frame=[], track_window=[])
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        rgb, _ = grab(camera)
        t0 = time.perf_counter()
        result = observe(rgb, predictor, profile=GREEN_PURPLE_PROFILE)
        t1 = time.perf_counter()
        square_candidates(rgb)
        t2 = time.perf_counter()
        cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        t3 = time.perf_counter()
        track_check(rgb, (160, 120))
        t4 = time.perf_counter()
        rows["balloon_full"].append((t1 - t0) * 1000)
        rows["balloon_search"].append(result["stage_ms"]["search"])
        rows["balloon_classify"].append(result["stage_ms"]["crop_and_classify"])
        rows["crops"].append(len(result["observations"]))
        rows["goal_squares"].append((t2 - t1) * 1000)
        rows["hsv_full_frame"].append((t3 - t2) * 1000)
        rows["track_window"].append((t4 - t3) * 1000)
    return {k: summary(v) for k, v in rows.items()}


def schedules(camera, predictor, seconds):
    def full(rgb, index):
        observe(rgb, predictor, profile=GREEN_PURPLE_PROFILE)
        square_candidates(rgb)

    def every(n):
        def work(rgb, index):
            if index % n == 0:
                full(rgb, index)
            else:
                track_check(rgb, (160, 120))
        return work

    out = {"full_every_frame": rate(camera, seconds, full)}
    for n in (3, 5, 10):
        out[f"full_1_in_{n}_track_between"] = rate(camera, seconds, every(n))
    out["track_only"] = rate(camera, seconds, lambda rgb, i: track_check(rgb, (160, 120)))
    return out


def zoom_delay(camera, factor, trials=5):
    """Frames from set_controls(ScalerCrop) until frame metadata reports the new crop."""
    native = camera.camera_properties["ScalerCropMaximum"]
    _, _, full_w, full_h = native
    w, h = int(full_w / factor) & ~1, int(full_h / factor) & ~1
    zoomed = (native[0] + (full_w - w) // 2, native[1] + (full_h - h) // 2, w, h)
    rows = []
    for trial in range(trials):
        for target in (zoomed, tuple(native)):
            camera.set_controls({"ScalerCrop": target})
            t0 = time.perf_counter()
            for frames in range(1, 31):
                _, metadata = grab(camera)
                crop = tuple(metadata.get("ScalerCrop", ()))
                if all(abs(a - b) <= 16 for a, b in zip(crop, target)) and crop:
                    rows.append(dict(to="zoom" if target == zoomed else "wide", frames=frames,
                                     ms=round((time.perf_counter() - t0) * 1000, 1), crop=list(crop)))
                    break
            else:
                rows.append(dict(to="zoom" if target == zoomed else "wide", frames=None,
                                 last_crop=list(crop)))
    return dict(factor=factor, requested=list(zoomed), native=list(native), trials=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seconds", type=float, default=10)
    args = parser.parse_args()
    if not 3 <= args.seconds <= 30:
        parser.error("Require 3..30 seconds per run")
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    predictor = Predictor(args.model, allow_unvalidated=True)
    report = dict(host=platform.node(), machine=platform.machine(), cpu_count=os.cpu_count(),
                  started=time.strftime("%Y-%m-%dT%H:%M:%S"), seconds_per_run=args.seconds,
                  model=Path(args.model).name, camera_only={}, stages=None, schedules=None,
                  zoom=None)

    for name, (sensor, fps) in SENSOR_MODES.items():
        camera = open_camera(sensor, fps)
        try:
            report["camera_only"][name] = dict(sensor=list(sensor), requested_fps=fps,
                                               **rate(camera, args.seconds))
            if name == "full":
                report["stages"] = stage_costs(camera, predictor, args.seconds)
                report["schedules"] = schedules(camera, predictor, args.seconds)
            if name == "binned":
                report["zoom"] = [zoom_delay(camera, 2), zoom_delay(camera, 4)]
                report["binned_schedules"] = schedules(camera, predictor, args.seconds)
        finally:
            camera.close()
        print(name, json.dumps(report["camera_only"][name]), flush=True)
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
