"""Bounded Pi Camera comparison; saves paired annotated frames, no flight commands.

This is visual inspection, not labeled accuracy. Both methods process each SAME
captured frame sequentially, so combined-loop FPS is not either method's live FPS.
"""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image

from dtr.data import write_json
from dtr.runtime import Predictor
from dtr.vision import observe
from pi_compare import load_baseline, verify_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", default=".")
    parser.add_argument("--output", required=True)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--task", choices=["goal", "balloon"], default="goal")
    parser.add_argument("--camera-rotation", type=int, choices=[0, 180], default=180)
    args = parser.parse_args()
    if not 1 <= args.frames <= 3000:
        parser.error("frames must be 1..3000")
    root, output = Path(args.bundle), Path(args.output)
    verify_bundle(root)
    if output.exists():
        raise FileExistsError(output)
    base = load_baseline(root / "baseline.py")
    detector = base.Detector(base.TARGET_COLOR_RGB, base.COLOR_THRESHOLD, base.MIN_SHAPE_SIZE, "numpy")
    predictor = Predictor(root / "models" / f"{args.task}.tflite", allow_unvalidated=True)
    from libcamera import Transform
    from picamera2 import Picamera2
    camera = Picamera2()
    output.mkdir(parents=True)
    try:
        camera.configure(camera.create_video_configuration(
            main={"size": (320, 240), "format": "RGB888"}, buffer_count=2,
            transform=Transform(hflip=args.camera_rotation==180, vflip=args.camera_rotation==180)))
        camera.start()
        time.sleep(1)
        with (output / "observations.jsonl").open("x") as stream:
            for index in range(args.frames):
                request = camera.capture_request()
                try:
                    # Picamera2 RGB888 is BGR in memory; inference requires RGB.
                    rgb = np.ascontiguousarray(request.make_array("main")[:, :, ::-1])
                    metadata = request.get_metadata()
                finally:
                    request.release()
                start = time.perf_counter()
                boxes, _ = detector.detect(rgb)
                baseline_ms = (time.perf_counter()-start)*1000
                result = observe(rgb, predictor, limit=12, profile="balloon_components", duplicate_policy="nested")
                stream.write(json.dumps(dict(frame=index, sensor_timestamp_ns=metadata.get("SensorTimestamp"),
                    baseline_ms=baseline_ms, baseline_boxes=boxes.tolist(), alternative=result,
                    flight_commands=None))+"\n")
                if index % 10 == 0 or index == args.frames-1:
                    left = base.draw_boxes(rgb, boxes, (255, 0, 0), 2)
                    right = rgb.copy()
                    for row in result["observations"]:
                        if row["accepted"]:
                            x1, y1, x2, y2 = map(int, row["box"])
                            cv2.rectangle(right, (x1,y1), (x2-1,y2-1), (0,255,0), 1)
                            cv2.putText(right, row["label"], (x1,max(24,y1)), cv2.FONT_HERSHEY_SIMPLEX, .3, (0,255,0), 1)
                    cv2.putText(left, "RGB baseline", (4,14), cv2.FONT_HERSHEY_SIMPLEX, .4, (255,255,255), 1)
                    cv2.putText(right, "INT8 student", (4,14), cv2.FONT_HERSHEY_SIMPLEX, .4, (255,255,255), 1)
                    Image.fromarray(rgb).save(output / f"{index:04}-raw.png")
                    Image.fromarray(np.concatenate([left,right], axis=1)).save(output / f"{index:04}-paired.png")
                    print(index+1, "frames", "baseline ms", round(baseline_ms,1),
                          "student ms", round(result["processing_ms"],1), flush=True)
        write_json(output / "summary.json", dict(frames=args.frames, task=args.task,
                   accuracy_measured=False, deployment_approved=False,
                   camera_rotation_degrees=args.camera_rotation,
                   coordinate_frame="orientation-corrected camera pixels",
                   note="Both methods share each camera frame; baseline remains yellow-only even in balloon mode"))
    finally:
        camera.close()


if __name__ == "__main__":
    cv2.setNumThreads(1)
    main()
