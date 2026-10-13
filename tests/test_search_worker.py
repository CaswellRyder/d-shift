import os
import time

import numpy as np
import pytest

from dtr.search_worker import SearchWorker


def fake_factory():
    def search(rgb, phase):
        if phase == "crash":
            os._exit(3)
        if phase == "error":
            raise RuntimeError("bad frame")
        return [dict(label="green_balloon", box=[0, 0, 1, 1], score=float(rgb[0, 0, 0]) / 255, phase=phase)]
    return search


def broken_factory():
    os._exit(4)  # Like a model that fails to load.


def wait_result(worker, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        result = worker.poll(wait=0.05)
        if result is not None:
            return result
    raise TimeoutError


@pytest.fixture
def worker():
    with SearchWorker(fake_factory, niceness=0, restart_delay=0.2) as w:
        w.start()
        w.wait_ready(20)
        yield w


def frame(value):
    return np.full((240, 320, 3), value, np.uint8)


def test_submit_returns_result_for_that_frame_and_refuses_while_busy(worker):
    assert worker.submit(frame(51), 12.5, "balloon")
    assert not worker.submit(frame(0), 12.6, "balloon")
    result = wait_result(worker)
    assert result["frame_time"] == 12.5 and result["phase"] == "balloon"
    assert result["detections"][0]["score"] == pytest.approx(0.2)
    assert worker.ready and worker.submit(frame(0), 13.0, "goal")
    assert wait_result(worker)["detections"][0]["phase"] == "goal"


def test_search_error_is_counted_and_worker_keeps_serving(worker):
    worker.submit(frame(1), 1.0, "error")
    end = time.monotonic() + 10
    while worker.errors == 0 and time.monotonic() < end:
        worker.poll(wait=0.05)
    assert worker.errors == 1 and "bad frame" in worker.last_error and worker.ready
    worker.submit(frame(1), 2.0, "balloon")
    assert wait_result(worker)["frame_time"] == 2.0


def test_dead_child_is_restarted_without_blocking(worker):
    worker.submit(frame(1), 1.0, "crash")
    end = time.monotonic() + 10
    while worker.restarts == 0 and time.monotonic() < end:
        assert worker.poll(wait=0.05) is None
    assert worker.restarts == 1
    worker.wait_ready(20)
    worker.submit(frame(2), 3.0, "balloon")
    assert wait_result(worker)["frame_time"] == 3.0


def test_rejects_wrong_frame_shape(worker):
    with pytest.raises(ValueError):
        worker.submit(np.zeros((480, 640, 3), np.uint8), 0.0, "balloon")


def test_child_that_cannot_start_is_not_retried_forever():
    with SearchWorker(broken_factory, niceness=0, restart_delay=0.1, max_failed_starts=2) as worker:
        worker.start()
        with pytest.raises(RuntimeError, match="failed to start"):
            worker.wait_ready(30)
        assert worker.state == "failed" and worker.restarts == 1
        assert worker.poll() is None and not worker.submit(frame(0), 0.0, "balloon")
