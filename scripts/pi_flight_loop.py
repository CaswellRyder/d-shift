"""Bounded flight-vision loop: per-frame tracker + low-priority full search + mission messages.

Writes the Pi -> ESP32 messages to a JSONL log; nothing is sent to a flight
controller. Use --replay to run on stored frames off the Pi.
"""

import argparse
from collections import Counter
import glob
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image

from dtr.flight import ColorTracker, Mission, encode
from dtr.goal_square import GOAL_COLORS
from dtr.search_worker import ModelSearch, SearchWorker

SENSOR = (1296, 972)  # 2x2 binned, full field of view.


class Camera:
    def __init__(self, fps):
        from libcamera import Transform
        from picamera2 import Picamera2

        self.camera = Picamera2()
        try:
            self.camera.configure(self.camera.create_video_configuration(
                main={"size": (320, 240), "format": "RGB888"},
                sensor={"output_size": SENSOR, "bit_depth": 10},
                controls={"FrameRate": fps},
                buffer_count=3,
                queue=False,
                transform=Transform(hflip=True, vflip=True),
            ))
            self.camera.start()
        except BaseException:
            self.camera.close()
            raise
        time.sleep(1)

    def read(self):
        request = self.camera.capture_request()
        try:
            rgb = cv2.cvtColor(request.make_array("main"), cv2.COLOR_BGR2RGB)  # RGB888 is BGR in memory.
        finally:
            request.release()
        return rgb, time.monotonic()

    def close(self):
        self.camera.close()


class Replay:
    def __init__(self, pattern, fps):
        paths = sorted(glob.glob(pattern))
        if not paths:
            raise ValueError("No replay frames")
        self.frames = [np.asarray(Image.open(p).convert("RGB").resize((320, 240))) for p in paths]
        self.period, self.index, self.next = 1 / fps, 0, time.monotonic()

    def read(self):
        self.next += self.period
        time.sleep(max(0.0, self.next - time.monotonic()))
        rgb = self.frames[self.index % len(self.frames)]
        self.index += 1
        return rgb, time.monotonic()

    def close(self):
        pass


def summary(values):
    values = np.asarray(values, float)
    if not len(values):
        return None
    return dict(mean=round(float(values.mean()), 1), p50=round(float(np.percentile(values, 50)), 1),
                p90=round(float(np.percentile(values, 90)), 1), max=round(float(values.max()), 1))


def run(source, worker, tracker, mission, seconds):
    lines, loop_times, refresh_times, search_ms, lag_ms = [], [], [], [], []
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        rgb, now = source.read()
        tracker.update(rgb, now)
        result = worker.poll()
        if result is not None:
            if result["phase"] == mission.search_phase:  # Ignore results from a finished phase.
                tracker.merge(result["detections"], result["frame_time"], now)
            refresh_times.append(now)
            search_ms.append(result["search_ms"])
            lag_ms.append((now - result["frame_time"]) * 1000)
        phase = mission.phase
        message = mission.step(tracker.tracks, now)
        if mission.phase != phase:
            tracker.clear()  # Each phase follows different objects.
        worker.submit(rgb, now, mission.search_phase)
        lines.append(encode(message))
        loop_times.append(time.monotonic())
    return lines, dict(
        frames=len(loop_times),
        loop_fps=round((len(loop_times) - 1) / (loop_times[-1] - loop_times[0]), 2),
        loop_gap_ms=summary(np.diff(loop_times) * 1000),
        search_results=len(refresh_times),
        search_refresh_ms=summary(np.diff(refresh_times) * 1000),
        search_ms=summary(search_ms),
        result_lag_ms=summary(lag_ms),
        worker_restarts=worker.restarts, worker_errors=worker.errors,
        commands=dict(Counter(json.loads(line)["cmd"] for line in lines)),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--balloon-model", required=True)
    parser.add_argument("--goal-model", required=True)
    parser.add_argument("--goal-color", required=True, choices=GOAL_COLORS)
    parser.add_argument("--phase", choices=("balloon", "goal"), default="balloon",
                        help="start phase; goal acts as if a balloon is already held")
    parser.add_argument("--log", required=True, help="JSONL message log to create")
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--fps", type=float, default=20)
    parser.add_argument("--replay", help="glob of stored frames instead of the camera")
    parser.add_argument("--niceness", type=int, default=19)
    args = parser.parse_args()
    if not 2 <= args.seconds <= 600 or not 1 <= args.fps <= 40:
        parser.error("Require seconds 2..600 and fps 1..40")
    log = Path(args.log)
    if log.exists():
        raise FileExistsError(log)

    mission = Mission(args.goal_color)
    if args.phase == "goal":
        mission.captured()
    worker = SearchWorker(ModelSearch(args.balloon_model, args.goal_model), niceness=args.niceness)
    try:
        worker.start()
        started = time.monotonic()
        worker.wait_ready()
        ready_s = time.monotonic() - started
        source = Replay(args.replay, args.fps) if args.replay else Camera(args.fps)
        try:
            lines, report = run(source, worker, ColorTracker(), mission, args.seconds)
        finally:
            source.close()
    finally:
        worker.close()
    log.write_text("".join(lines))  # Written after the loop: no disk I/O while timing.
    report.update(worker_ready_s=round(ready_s, 2), source="replay" if args.replay else "camera",
                  fps=args.fps, goal_color=args.goal_color, start_phase=args.phase, log=str(log))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
