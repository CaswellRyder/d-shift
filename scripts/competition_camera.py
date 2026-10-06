"""Bounded live throughput trials, pixel/context/context/pixel. No control output.

Scenes are unlabeled and change between passes: do not derive comparative accuracy.
"""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np

from competition_compare import load_pixel
from dtr.data import sha256
from dtr.runtime import Predictor
from dtr.temporal import TemporalVision


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pixel", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seconds", type=float, default=10)
    p.add_argument("--camera-rotation", type=int, choices=[0, 180], default=180)
    args = p.parse_args()
    if not 3 <= args.seconds <= 30:
        p.error("seconds must be 3..30 per pass")
    out = Path(args.output)
    out.mkdir(exist_ok=False)
    cv2.setNumThreads(1)
    pixel = load_pixel(args.pixel)
    predictor = Predictor(args.model, allow_unvalidated=True)
    from libcamera import Transform
    from picamera2 import Picamera2
    camera = Picamera2()
    report = dict(accuracy_measured=False, flight_commands=None,
                  source_size=[640,480], source_fps_requested=10,
                  pixel_search_size=[640,480], context_search_size=[320,240],
                  camera_rotation_degrees=args.camera_rotation,
                  coordinate_frame="orientation-corrected camera pixels",
                  model_sha256=sha256(args.model), pixel_sha256=sha256(args.pixel),
                  scope="Native processing sizes, separate unlabeled live passes; not same-scene accuracy. "
                        "Sensor age uses CLOCK_BOOTTIME minus SensorTimestamp; negative ages rejected. "
                        "Per-frame records retained in memory; PNG and JSON writes occur after timed passes.",
                  passes=[])
    samples = []
    try:
        camera.configure(camera.create_video_configuration(
            main={"size":(640,480), "format":"RGB888"}, buffer_count=2,
            controls={"FrameRate":10}, queue=False,
            transform=Transform(hflip=args.camera_rotation==180, vflip=args.camera_rotation==180)))
        report["configured_transform"] = str(camera.camera_configuration()["transform"])
        camera.start()
        time.sleep(1)
        for pass_id, method in enumerate(["pixel", "context", "context", "pixel"]):
            memory = pixel.GoalMemory()
            tracker = TemporalVision(predictor, budget=4, difference_backend="native")
            records = []
            start = time.monotonic()
            while time.monotonic()-start < args.seconds:
                request = camera.capture_request()
                try:
                    bgr = request.make_array("main").copy()
                    metadata = request.get_metadata()
                finally:
                    request.release()
                operation = time.perf_counter()
                if method == "pixel":
                    goal, _ = memory.update(bgr)
                    boxes = [] if goal is None else [list(goal["box"])]
                else:
                    rgb = cv2.cvtColor(cv2.resize(bgr,(320,240)), cv2.COLOR_BGR2RGB)
                    result = tracker.observe(rgb)
                    boxes = [[int(v*2) for v in o["box"]] for o in result["observations"]
                             if o["accepted"] and o["label"].startswith("yellow_")]
                process_ms = (time.perf_counter()-operation)*1000
                end_ns = time.clock_gettime_ns(time.CLOCK_BOOTTIME)
                sensor = metadata.get("SensorTimestamp")
                age = (end_ns-sensor)/1e6 if sensor else None
                if age is not None and age < 0:
                    raise ValueError("Incompatible sensor timestamp clock")
                records.append(dict(sensor_timestamp_ns=sensor, result_boottime_ns=end_ns,
                                    sensor_to_result_ms=age, processing_ms=process_ms,
                                    yellow_boxes=boxes, exposure_us=metadata.get("ExposureTime"),
                                    analogue_gain=metadata.get("AnalogueGain")))
                if len(records) == 1:
                    samples.append((f"{pass_id}-{method}-first", bgr.copy(), boxes))
            elapsed = time.monotonic()-start
            samples.append((f"{pass_id}-{method}-last", bgr.copy(), boxes))
            times = [r["processing_ms"] for r in records]
            ages = [r["sensor_to_result_ms"] for r in records if r["sensor_to_result_ms"] is not None]
            stamps = [r["sensor_timestamp_ns"] for r in records]
            item = dict(method=method, frames=len(records), elapsed_s=elapsed,
                        live_fps=len(records)/elapsed, mean_processing_ms=float(np.mean(times)),
                        p95_processing_ms=float(np.percentile(times,95)),
                        p95_sensor_to_result_ms=float(np.percentile(ages,95)) if ages else None,
                        sensor_timestamps_increasing=all(a is not None and b is not None and b>a
                                                       for a,b in zip(stamps,stamps[1:])),
                        emitted_yellow_boxes=sum(len(r["yellow_boxes"]) for r in records), records=records)
            report["passes"].append(item)
            print({k:v for k,v in item.items() if k != "records"}, flush=True)
    finally:
        camera.close()
    for name, bgr, boxes in samples:
        cv2.imwrite(str(out/f"{name}-raw.png"), bgr)
        canvas = bgr.copy()
        for a,b,c,d in boxes:
            cv2.rectangle(canvas,(int(a),int(b)),(int(c),int(d)),(0,255,0),2)
        cv2.imwrite(str(out/f"{name}-annotated.png"), canvas)
    (out/"report.json").write_text(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
