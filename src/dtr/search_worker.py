"""Full search in a low-priority child process, so the per-frame tracker never waits.

On the single-core Pi Zero, `nice 19` lets the kernel pause the search whenever
the tracker has a frame. The caller never blocks: submit() refuses while busy
and poll() returns a finished result or None. If the child dies, poll()
restarts it in the background.
"""

import multiprocessing as mp
from multiprocessing import shared_memory
import os
import sys
import time

import numpy as np

FRAME_SHAPE = (240, 320, 3)


def _serve(conn, shm_name, shape, factory, niceness):
    if niceness:
        os.nice(niceness)
    shm = shared_memory.SharedMemory(name=shm_name)
    frame = np.ndarray(shape, np.uint8, buffer=shm.buf)
    try:
        search = factory()
        search(np.zeros(shape, np.uint8), "balloon")  # Load and warm up before reporting ready.
        conn.send(("ready",))
        while True:
            message = conn.recv()
            if message is None:
                break
            frame_time, phase = message
            rgb = frame.copy()
            start = time.perf_counter()
            try:
                detections = search(rgb, phase)
            except Exception as error:  # Report and keep serving; the tracker decides what to do.
                conn.send(("error", frame_time, phase, repr(error)))
                continue
            conn.send(("result", frame_time, phase, detections, (time.perf_counter() - start) * 1000))
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        del frame
        shm.close()


class SearchWorker:
    """`factory()` runs in the child and returns `search(rgb, phase) -> [detection, ...]`.

    Linux forks (fast, the Pi). Elsewhere the child is spawned, so `factory` must be
    picklable; macOS cannot fork safely once TensorFlow has started.
    """

    def __init__(self, factory, shape=FRAME_SHAPE, niceness=19, restart_delay=1.0, max_failed_starts=3):
        self.factory, self.shape, self.niceness = factory, tuple(shape), niceness
        self.context = mp.get_context("fork" if sys.platform.startswith("linux") else "spawn")
        self.shm = shared_memory.SharedMemory(create=True, size=int(np.prod(self.shape)))
        self.frame = np.ndarray(self.shape, np.uint8, buffer=self.shm.buf)
        self.state, self.process, self.conn = "stopped", None, None
        self.restarts, self.errors, self.last_error = 0, 0, None
        self.restart_delay, self.max_failed_starts = restart_delay, max_failed_starts
        self.failed_starts, self.died_at = 0, None

    def start(self):
        self.conn, child = self.context.Pipe()
        self.process = self.context.Process(
            target=_serve, args=(child, self.shm.name, self.shape, self.factory, self.niceness),
            daemon=True)
        self.process.start()
        child.close()
        self.state = "starting"

    def wait_ready(self, timeout=60.0):
        """Block until the child has loaded its models. Call before flight, not in the loop."""
        end = time.monotonic() + timeout
        while self.state != "idle":
            if self.state == "failed":
                raise RuntimeError(f"Search worker failed to start: {self.last_error}")
            if time.monotonic() > end:
                raise TimeoutError("Search worker did not become ready")
            self.poll(wait=min(0.1, max(0.0, end - time.monotonic())))
        return self

    @property
    def ready(self):
        return self.state == "idle"

    def submit(self, rgb, frame_time, phase):
        """Hand the child a frame. Returns False (and does nothing) unless it is idle."""
        if self.state != "idle":
            return False
        if rgb.shape != self.shape or rgb.dtype != np.uint8:
            raise ValueError("Frame does not match the worker's shared buffer")
        self.frame[:] = rgb  # Safe: the child only reads the buffer while busy.
        self.conn.send((frame_time, phase))
        self.state = "busy"
        return True

    def poll(self, wait=0.0):
        """A finished result dict, or None. Restarts a dead child after a short delay."""
        if self.state == "failed":
            return None
        if self.state == "dead":
            if time.monotonic() - self.died_at >= self.restart_delay:
                self.restarts += 1
                self.start()
            return None
        if self.process is None:
            return None
        try:
            ready = self.conn.poll(wait)
            message = self.conn.recv() if ready else None
        except (EOFError, OSError):
            message = None
        if message is None:
            if not self.process.is_alive():
                self._died()
            return None
        kind = message[0]
        if kind == "ready":
            self.state, self.failed_starts = "idle", 0
            return None
        self.state = "idle"
        if kind == "error":
            self.errors += 1
            self.last_error = message[3]
            return None
        _, frame_time, phase, detections, search_ms = message
        return dict(frame_time=frame_time, phase=phase, detections=detections, search_ms=search_ms)

    def _died(self):
        self.process.join(0.1)
        self.conn.close()
        if self.state == "starting":
            self.failed_starts += 1
            self.last_error = f"child exited with code {self.process.exitcode} before ready"
        self.process, self.died_at = None, time.monotonic()
        # A child that cannot even load is not retried forever.
        self.state = "failed" if self.failed_starts >= self.max_failed_starts else "dead"

    def close(self):
        if self.process is not None:
            try:
                self.conn.send(None)
            except (BrokenPipeError, OSError):
                pass
            self.process.join(2)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(2)
            self.conn.close()
            self.process = None
        self.state = "stopped"
        if self.shm is not None:
            del self.frame
            self.shm.close()
            self.shm.unlink()
            self.shm = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class ModelSearch:
    """Picklable factory for the real models; loads them inside the child process."""

    def __init__(self, balloon_model, goal_model, balloon_limit=8):
        self.balloon_model, self.goal_model = str(balloon_model), str(goal_model)
        self.balloon_limit = balloon_limit

    def __call__(self):
        from .goal_square import find_squares
        from .runtime import Predictor
        from .vision import GREEN_PURPLE_PROFILE, observe

        balloons = Predictor(self.balloon_model, allow_unvalidated=True)
        goals = Predictor(self.goal_model, allow_unvalidated=True)
        balloon_limit = self.balloon_limit

        def search(rgb, phase):
            if phase == "goal":
                found = find_squares(rgb, goals)["detections"]  # Ours and the opponent's.
                return [dict(label=d["label"], box=[float(v) for v in d["box"]], score=d["score"])
                        for d in found]
            found = observe(rgb, balloons, limit=balloon_limit, profile=GREEN_PURPLE_PROFILE)
            return [dict(label=d["label"], box=[float(v) for v in d["box"]], score=float(d["score"]))
                    for d in found["observations"] if d["accepted"] and not d["suppressed"]]

        return search
