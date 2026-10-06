import io
import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest
from PIL import Image

from dtr.tracking import BoxTracker, iou
from dtr.web import VisionService, make_server


def test_tracker_matches_once_and_expires():
    tracker = BoxTracker()
    rows = [{"box": [10, 10, 30, 30]}, {"box": [100, 100, 120, 120]}]
    first = tracker.update(rows, 1)
    second = tracker.update([rows[1], rows[0]], 1.1)
    assert [r["track_id"] for r in first] == [1, 2]
    assert [r["track_id"] for r in second] == [2, 1]
    assert len(second[0]["trail"]) == 2
    assert tracker.update(rows, 3)[0]["track_id"] == 3
    assert iou(rows[0]["box"], rows[0]["box"]) == 1
    assert iou(rows[0]["box"], rows[1]["box"]) == 0


@pytest.fixture
def service(monkeypatch):
    class FakePredictor:
        metadata = {
            "task": "balloon",
            "classes": ["background", "green_balloon"],
            "synthetic_training": True,
            "threshold": 0.8,
            "deployment_approved": False,
        }

        def __init__(self, path, allow):
            assert allow

        def predict(self, rgb):
            return {
                "label": "green_balloon",
                "score": 0.5,
                "scores": [0.5, 0.5],
                "accepted": False,
                "synthetic_training": True,
                "deployment_approved": False,
            }

    monkeypatch.setattr("dtr.web.Predictor", FakePredictor)
    return VisionService({"balloon": "fixture"}, allow_unvalidated=True)


def jpeg(size=(320, 240)):
    rgb = np.zeros((size[1], size[0], 3), np.uint8)
    rgb[40:100, 40:100] = [30, 220, 30]
    out = io.BytesIO()
    Image.fromarray(rgb).save(out, format="JPEG")
    return out.getvalue()


def test_service_tracks_and_rejects_stale_frames(service):
    a = service.process(jpeg(), "balloon", "session1", 0)
    b = service.process(jpeg(), "balloon", "session1", 1)
    assert a["observations"][0]["track_id"] == b["observations"][0]["track_id"]
    assert len(b["observations"][0]["trail"]) == 2
    assert b["mask"].startswith("data:image/png;base64,")
    assert b["flight_commands"] is None and not b["deployment_approved"]
    with pytest.raises(ValueError, match="Stale"):
        service.process(jpeg(), "balloon", "session1", 1)
    with pytest.raises(ValueError, match="320x240"):
        service.process(jpeg((640, 480)), "balloon", "session1", 2)
    with pytest.raises(ValueError, match="Unknown"):
        service.process(jpeg(), "goal", "session1", 2)
    for seq in range(2, 12):
        restarted = service.process(jpeg(), "balloon", "session1", seq, reset=True)
        assert len(restarted["observations"][0]["trail"]) == 1
    assert len(service.sessions) == 1


def test_service_goal_budget_is_bounded_and_does_not_change_balloon_budget(service):
    assert service.config()["tasks"]["balloon"]["proposal_limit"] == 12
    assert service.config()["proposal_profile"] == "balloon_components"
    with pytest.raises(ValueError, match="Viewer proposal profile"):
        VisionService({}, proposal_profile="orange_local")
    assert VisionService({}, goal_proposal_limit=24).limits == {"balloon": 12, "goal": 24}
    for bad in (0, 13, 65, True):
        with pytest.raises(ValueError, match="Goal proposal limit"):
            VisionService({}, goal_proposal_limit=bad)


def test_service_limits_sessions(service):
    for i in range(8):
        service.process(jpeg(), "balloon", f"session{i}", 0)
    with pytest.raises(ValueError, match="Too many"):
        service.process(jpeg(), "balloon", "another", 0)


def test_local_server_api_and_boundaries(service):
    server = make_server(service, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(base) as response:
            html = response.read()
            assert b"DTR Vision" in html
            assert b'/frame-policy.js' in html and b'id="fps"' in html
            assert b'/goal-view.js' in html and b'Tentative shape-only' in html
            assert response.headers["Cache-Control"] == "no-store"
        with urllib.request.urlopen(base + "/frame-policy.js") as response:
            assert b"SOURCE_WIDTH = 2592, SOURCE_HEIGHT = 1944" in response.read()
        with urllib.request.urlopen(base + "/goal-view.js") as response:
            assert b"DTRGoalView" in response.read()
            assert response.headers["Cache-Control"] == "no-store"
        with urllib.request.urlopen(base + "/api/config") as response:
            config = json.load(response)
        endpoint = base + "/api/frame?task=balloon&session=test&sequence=0"
        headers = {"Content-Type": "image/jpeg", "X-DTR-Token": config["token"]}
        request = urllib.request.Request(endpoint, jpeg(), headers)
        with urllib.request.urlopen(request) as response:
            assert json.load(response)["observations"][0]["track_id"] == 1
        for bad_headers in [
            {"X-DTR-Token": "bad"},
            {"Origin": "https://example.com"},
            {"Host": "attacker.invalid"},
        ]:
            request = urllib.request.Request(endpoint, jpeg(), {**headers, **bad_headers})
            with pytest.raises(urllib.error.HTTPError) as exc:
                urllib.request.urlopen(request)
            assert exc.value.code == 403
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(base + "/../../pyproject.toml")
        assert exc.value.code == 404
        request = urllib.request.Request(endpoint, b"not-an-image", headers)
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(request)
        assert exc.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_explicit_private_https_proxy_origin(service):
    origin = "https://pacman.example.ts.net:8765"
    host = "pacman.example.ts.net:8765"
    server = make_server(service, port=0, public_origins=[origin])
    assert server.server_address[0] == "127.0.0.1"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        request = urllib.request.Request(base + "/api/config", headers={"Host": host})
        with urllib.request.urlopen(request) as response:
            token = json.load(response)["token"]
        headers = {"Host": host, "Origin": origin, "X-DTR-Token": token,
                   "Content-Type": "image/jpeg", "Sec-Fetch-Site": "same-origin"}
        endpoint = base + "/api/frame?task=balloon&session=proxy&sequence=0"
        with urllib.request.urlopen(urllib.request.Request(endpoint, jpeg(), headers)) as response:
            assert response.status == 200
        for bad in ({"Origin": "https://attacker.invalid"}, {"Host": "attacker.invalid"},
                    {"Origin": origin.replace("https:", "http:")},
                    {"Sec-Fetch-Site": "cross-site"}, {"X-DTR-Token": "bad"},
                    {"Host": f"127.0.0.1:{server.server_port}"}):
            with pytest.raises(urllib.error.HTTPError) as exc:
                urllib.request.urlopen(urllib.request.Request(endpoint, jpeg(), {**headers, **bad}))
            assert exc.value.code == 403
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize("origin", ["http://pacman:8765", "https://*.example.com",
                                  "https://user:pass@example.com", "https://example.com/path",
                                  "https://example.com?x=1", "https://example.com#x"])
def test_reject_invalid_public_origins(service, origin):
    with pytest.raises(ValueError):
        make_server(service, port=0, public_origins=[origin])
