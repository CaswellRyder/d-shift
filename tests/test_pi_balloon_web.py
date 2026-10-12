import json
import threading
import time

import numpy as np
import pytest

from scripts import pi_balloon_web as web


def row(*labels):
    return dict(detections=[dict(label=label, accepted=accepted, box=[0, 0, 4, 4])
                            for label, accepted in labels])


def test_recorder_rejects_bad_input(tmp_path):
    recorder = web.Recorder(tmp_path)
    for args in [("Bad Name", 5, 1, 0, ""), ("../up", 5, 1, 0, ""), ("ok", 121, 1, 0, ""),
                 ("ok", 5, -1, 0, ""), ("ok", 5, 0, 21, ""), ("ok", 5, 0, 0, "x" * 301)]:
        with pytest.raises(ValueError):
            recorder.start(*args, context={})
    assert recorder.active is None
    assert not any(p.is_dir() for p in tmp_path.rglob("*-*"))


def test_recorder_writes_frames_rows_and_counts(tmp_path):
    recorder = web.Recorder(tmp_path)
    recorder.start("red-2m", 60, 1, 0, "lit bench", context=dict(model="m"))
    with pytest.raises(ValueError):
        recorder.start("red-2m", 60, 1, 0, "", context={})
    rgb = np.zeros((240, 320, 3), np.uint8)
    recorder.write(rgb, row(("red_balloon", True), ("blue_balloon", False)))
    recorder.write(rgb, row(("red_balloon", True), ("background", False)))
    folder = recorder.active["folder"]
    recorder.stop()
    assert folder.name == "red-2m-01"
    assert sorted(p.name for p in folder.glob("*.png")) == ["0000.png", "0001.png"]
    rows = [json.loads(line) for line in (folder / "frames.jsonl").read_text().splitlines()]
    assert [r["path"] for r in rows] == ["0000.png", "0001.png"]
    meta = json.loads((folder / "scene.json").read_text())
    assert meta["frames"] == 2 and meta["model"] == "m" and meta["training_use"] is False
    assert meta["accepted_detections"] == {"red_balloon": 2, "blue_balloon": 0}
    assert recorder.status()["last"]["frames"] == 2
    recorder.start("red-2m", 0, 1, 0, "", context={})
    recorder.write(rgb, row())
    assert recorder.active is None and recorder.last["folder"].endswith("red-2m-02")


def test_viewer_publishes_frames_and_stops_cleanly(tmp_path):
    class Source:
        closed = False

        def read(self):
            time.sleep(0.01)
            return np.zeros((240, 320, 3), np.uint8), {"SensorTimestamp": web.boottime_ns()}

        def close(self):
            self.closed = True

    def infer(rgb, policy):
        return dict(row(("blue_balloon", True)), processing_ms=1.0, search_ms=0.5,
                    inference_ms=0.4, neural_calls=1)

    source = Source()
    viewer = web.Viewer(source, infer, web.Recorder(tmp_path), {})
    viewer.watched = time.monotonic()
    thread = threading.Thread(target=viewer.loop)
    thread.start()
    state = viewer.state(0)
    later = viewer.state(state["sequence"])
    viewer.running = False
    thread.join(timeout=2)
    assert later["sequence"] > state["sequence"] and later["frame"]
    assert later["row"]["detections"][0]["label"] == "blue_balloon"
    assert later["error"] is None and source.closed


def test_host_check_accepts_only_loopback_names():
    for host in ("localhost:8765", "127.0.0.1:9000"):
        assert web.LOOPBACK_HOST.fullmatch(host)
    for host in ("vision.local:8765", "evil.example:8765", "localhost", "localhost.evil:80"):
        assert not web.LOOPBACK_HOST.fullmatch(host)


def test_recovery_runs_pixel_search_between_full_searches():
    from dtr.research_search_schedule import SearchSchedule
    clock = iter(range(0, 10**10, 10**7))
    scheduler, calls = SearchSchedule(), []

    def search(name, fail=False):
        def run(rgb):
            calls.append(name)
            if fail and len(calls) == 2:
                raise RuntimeError("camera glitch")
            return dict(detections=[])
        return run

    rows = [web.recovery_frame(None, search("full"), search("pixel"), scheduler, lambda: next(clock))
            for _ in range(7)]
    assert [r["mode"] for r in rows] == ["mser_direct", "pixel2", "pixel2", "mser_direct",
                                         "pixel2", "pixel2", "mser_direct"]
    assert calls == ["full", "pixel", "pixel", "full", "pixel", "pixel", "full"]
    assert rows[3]["reason"] == "frame_cap"

    scheduler, calls = SearchSchedule(), []
    fast = search("pixel", fail=True)
    web.recovery_frame(None, search("full"), fast, scheduler, lambda: next(clock))
    with pytest.raises(RuntimeError):
        web.recovery_frame(None, search("full"), fast, scheduler, lambda: next(clock))
    assert web.recovery_frame(None, search("full"), fast, scheduler, lambda: next(clock))["mode"] == "mser_direct"


def test_policy_endpoint_offers_recovery_only_with_pixel_golden(tmp_path):
    import http.client

    def post(port, token, policy):
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("POST", "/api/policy", json.dumps(dict(policy=policy)),
                     {"X-Token": token, "Content-Type": "application/json"})
        response = conn.getresponse()
        response.read()
        conn.close()
        return response.status

    for policies in (web.POLICIES[:2], web.POLICIES):
        viewer = web.Viewer(None, None, web.Recorder(tmp_path), dict(policies=policies))
        server = web.make_server(viewer, 0)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            port = server.server_address[1]
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            conn.request("GET", "/api/config")
            token = json.loads(conn.getresponse().read())["token"]
            conn.close()
            assert post(port, token, "periodic") == 200 and viewer.policy == "periodic"
            assert post(port, token, "recovery") == (200 if "recovery" in policies else 400)
            assert viewer.policy == ("recovery" if "recovery" in policies else "periodic")
            assert post(port, token, "fast") == 400
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
