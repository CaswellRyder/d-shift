"""Probe ways to keep per-frame tracking alive while the full search runs.

The live camera paces the loop; processing runs on stored lit arena frames so
results do not depend on room lighting. Timing only, no flight commands.
"""

import argparse
import glob
import json
import multiprocessing as mp
from multiprocessing import shared_memory
import os
import threading
import time

import numpy as np
from PIL import Image

from dtr.goal_square import GOAL_COLORS, square_candidates
from dtr.runtime import Predictor
from dtr.vision import GREEN_PURPLE_PROFILE, proposals, suppress_duplicates

import pi_fps_bench as bench

LIMIT = 8  # Proposal limit the green/purple golden uses.
SHAPE = (240, 320, 3)


def full_job(rgb, predictor, balloons=True, goals=True):
    """Full search as resumable steps; each yield is a safe place to hand back the CPU."""
    found = []
    if balloons:
        candidates = proposals(rgb, "balloon", LIMIT, profile=GREEN_PURPLE_PROFILE)
        yield "search"
        for start in range(0, len(candidates), 2):
            for c in candidates[start:start + 2]:
                x1, y1, x2, y2 = c["crop_box"]
                found.append({**c, **predictor.predict(rgb[y1:y2, x1:x2])})
            yield "classify"
        found = suppress_duplicates(found)
    if goals:
        for color in GOAL_COLORS:
            square_candidates(rgb, (color,))
            yield "goal_" + color
    return found


def run_blocking(predictor, rgb):
    for _ in full_job(rgb, predictor):
        pass


class Loop:
    """Foreground loop: grab, cheap track check, then whatever the strategy adds."""

    def __init__(self, camera, frames):
        self.camera, self.frames = camera, frames
        self.updates, self.refreshes, self.latency = [], [], []

    def run(self, seconds, after_track):
        end = time.perf_counter() + seconds
        index = 0
        while time.perf_counter() < end:
            bench.grab(self.camera)
            rgb = self.frames[index % len(self.frames)]
            bench.track_check(rgb, (160, 120))
            self.updates.append(time.perf_counter())
            after_track(index, rgb)
            index += 1

    def done(self, snapshot_at):
        now = time.perf_counter()
        self.refreshes.append(now)
        self.latency.append((now - snapshot_at) * 1000)

    def report(self):
        gaps = np.diff(self.updates) * 1000
        return dict(track_fps=round((len(self.updates) - 1) / (self.updates[-1] - self.updates[0]), 2),
                    track_gap_ms=bench.summary(gaps),
                    full_results=len(self.refreshes),
                    full_refresh_ms=bench.summary(np.diff(self.refreshes) * 1000),
                    full_latency_ms=bench.summary(self.latency))


def strategy_blocking(loop, predictor, n):
    def after(index, rgb):
        if index % n == 0:
            t0 = time.perf_counter()
            run_blocking(predictor, rgb)
            loop.done(t0)
    return after


def strategy_sliced(loop, predictor, budget_ms, balloons=True, goals=True):
    state = dict(job=None, started=None)

    def after(index, rgb):
        frame_start = loop.updates[-1]
        while True:
            if state["job"] is None:
                state["job"] = full_job(rgb.copy(), predictor, balloons, goals)
                state["started"] = time.perf_counter()
            try:
                next(state["job"])
            except StopIteration:
                loop.done(state["started"])
                state["job"] = None
            if (time.perf_counter() - frame_start) * 1000 >= budget_ms:
                return
    return after


def worker_process(conn, shm_name, model, niceness, balloons=True, goals=True):
    os.nice(niceness)
    predictor = Predictor(model, allow_unvalidated=True)
    shm = shared_memory.SharedMemory(name=shm_name)
    frame = np.ndarray(SHAPE, np.uint8, buffer=shm.buf)
    try:
        for _ in full_job(frame.copy(), predictor, balloons, goals):
            pass  # Warm up before reporting ready.
        conn.send("ready")
        while conn.recv():
            for _ in full_job(frame.copy(), predictor, balloons, goals):
                pass
            conn.send(True)
    finally:
        del frame
        shm.close()


def with_process(loop, model, niceness, seconds, balloons=True, goals=True):
    shm = shared_memory.SharedMemory(create=True, size=int(np.prod(SHAPE)))
    frame = np.ndarray(SHAPE, np.uint8, buffer=shm.buf)
    parent, child = mp.get_context("fork").Pipe()
    proc = mp.get_context("fork").Process(target=worker_process, args=(child, shm.name, model, niceness, balloons, goals))
    proc.start()
    if not parent.poll(60) or parent.recv() != "ready":
        raise RuntimeError("Search worker did not start")
    state = dict(busy=False, started=None)

    def after(index, rgb):
        if state["busy"] and parent.poll():
            parent.recv()
            loop.done(state["started"])
            state["busy"] = False
        if not state["busy"]:
            frame[:] = rgb
            parent.send(True)
            state.update(busy=True, started=time.perf_counter())

    try:
        loop.run(seconds, after)
    finally:
        if state["busy"]:
            parent.recv()
        parent.send(False)
        proc.join(5)
        frame = None  # Release the buffer view before closing the segment.
        shm.close()
        shm.unlink()


def with_thread(loop, predictor, seconds):
    jobs, results = [], []
    wake = threading.Event()
    stop = threading.Event()

    def work():
        while not stop.is_set():
            if not wake.wait(0.1):
                continue
            wake.clear()
            rgb, started = jobs.pop()
            run_blocking(predictor, rgb)
            results.append(started)

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    state = dict(busy=False)

    def after(index, rgb):
        if results:
            loop.done(results.pop())
            state["busy"] = False
        if not state["busy"]:
            jobs.append((rgb.copy(), time.perf_counter()))
            wake.set()
            state["busy"] = True

    try:
        loop.run(seconds, after)
    finally:
        stop.set()
        thread.join(5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--frames", required=True, help="glob of stored lit arena frames")
    parser.add_argument("--seconds", type=float, default=12)
    parser.add_argument("--fps", type=float, default=20)
    parser.add_argument("--only", help="run just the measurements whose name starts with this")
    args = parser.parse_args()
    frames = [np.asarray(Image.open(p).convert("RGB").resize((320, 240)))
              for p in sorted(glob.glob(args.frames))]
    if not frames:
        raise ValueError("No frames")
    predictor = Predictor(args.model, allow_unvalidated=True)
    camera = bench.open_camera((1296, 972), args.fps)
    out = {}
    try:
        def measure(name, setup):
            if args.only and not name.startswith(args.only):
                return
            loop = Loop(camera, frames)
            setup(loop)
            out[name] = loop.report()
            print(name, json.dumps(out[name]), flush=True)

        measure("blocking_1_in_10", lambda loop: loop.run(args.seconds, strategy_blocking(loop, predictor, 10)))
        for budget in (20, 35):
            measure(f"sliced_budget_{budget}ms",
                    lambda loop: loop.run(args.seconds, strategy_sliced(loop, predictor, budget)))
        measure("sliced_35ms_balloons_only",
                lambda loop: loop.run(args.seconds, strategy_sliced(loop, predictor, 35, goals=False)))
        measure("sliced_35ms_goals_only",
                lambda loop: loop.run(args.seconds, strategy_sliced(loop, predictor, 35, balloons=False)))
        measure("thread", lambda loop: with_thread(loop, predictor, args.seconds))
        for niceness in (0, 10, 19):
            measure(f"process_nice_{niceness}", lambda loop: with_process(loop, args.model, niceness, args.seconds))
        measure("process_nice_19_repeat", lambda loop: with_process(loop, args.model, 19, args.seconds))
        measure("process_nice_19_balloons_only",
                lambda loop: with_process(loop, args.model, 19, args.seconds, goals=False))
        measure("process_nice_19_goals_only",
                lambda loop: with_process(loop, args.model, 19, args.seconds, balloons=False))
    finally:
        camera.close()
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
