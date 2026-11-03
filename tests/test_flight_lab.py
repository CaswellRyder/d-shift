"""Synthetic checks for flight-equivalent lab captures; never touch camera or Pi."""

from copy import deepcopy
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time

import cv2
import numpy as np
import pytest

from dtr.flight import ColorTracker, Mission, Track
from dtr.flight_lab import CaptureBuffer, export_capture, validate_capture
from dtr.flight_pipeline import FlightPipeline
from scripts import pi_flight_lab as lab_web


def payload():
    return dict(distance_m=1.2345, scene="measured-balloon", notes="Tape measured",
                target_label="green_balloon", expected_count=1)


def valid_metadata():
    return validate_capture(payload(), "balloon", "orange")


def rgb():
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.circle(image, (160, 120), 12, (40, 200, 60), -1)
    return image


def captured_row(when=0.0):
    return dict(frame_time=when, loop_completed=when + .002, phase_before="balloon",
                phase="balloon", tracks=[dict(id=1, label="green_balloon", score=.95,
                box=[148., 108., 172., 132.], confirmed=when, seen=when, misses=0)],
                message=dict(v=1, seq=1, t_ms=int(when * 1000), phase="balloon",
                goal_color="orange", cmd="TRACK", fan=False,
                target=dict(label="green_balloon", x=0., y=0., size=.075,
                            age_ms=0, seen_ms=0)), search_result=None, submitted=True)


class StubWorker:
    def __init__(self, results=()):
        self.results = list(results)
        self.submissions = []
        self.errors = self.restarts = 0
        self.ready = True
        self.state = "idle"

    def poll(self):
        return self.results.pop(0) if self.results else None

    def submit(self, image, when, phase):
        self.submissions.append((when, phase))
        return True

    def close(self):
        self.closed = True


def search_result(when, phase="balloon"):
    label = "green_balloon" if phase == "balloon" else "orange_square"
    return dict(frame_time=when, phase=phase, search_ms=250.,
                detections=[dict(label=label, score=.95, box=[148., 108., 172., 132.])])


def test_shared_pipeline_matches_legacy_processing_sequence_and_messages():
    """Compare with the flight runner's original ordering, using real tracker/mission."""
    results = [search_result(0.), None, None, search_result(.1), None, None]
    old_worker, worker = StubWorker(deepcopy(results)), StubWorker(deepcopy(results))
    old_tracker, tracker = ColorTracker(), ColorTracker()
    old_mission, mission = Mission("orange"), Mission("orange")
    pipeline = FlightPipeline(worker, tracker, mission)
    image = rgb()
    for when in np.arange(6) * .05:
        old_tracker.update(image, when)
        found = old_worker.poll()
        if found is not None and found["phase"] == old_mission.search_phase:
            old_tracker.merge(found["detections"], found["frame_time"], when)
        phase = old_mission.phase
        expected = old_mission.step(old_tracker.tracks, when)
        if old_mission.phase != phase:
            old_tracker.clear()
        old_worker.submit(image, when, old_mission.search_phase)
        actual = pipeline.step(image, when)
        assert actual["message"] == expected
        assert [t.box for t in tracker.tracks] == [t.box for t in old_tracker.tracks]
    assert worker.submissions == old_worker.submissions


def test_pipeline_preserves_update_poll_merge_mission_submit_order():
    events = []

    class Tracker:
        tracks = []

        def update(self, image, when):
            events.append("update")

        def merge(self, *args):
            events.append("merge")

        def clear(self):
            events.append("clear")

    class Worker(StubWorker):
        def poll(self):
            events.append("poll")
            return search_result(0.)

        def submit(self, *args):
            events.append("submit")
            return True

    class LoggedMission(Mission):
        def step(self, tracks, when):
            events.append("mission")
            return super().step(tracks, when)

    FlightPipeline(Worker(), Tracker(), LoggedMission("orange")).step(rgb(), .1)
    assert events == ["update", "poll", "merge", "mission", "submit"]


def test_reentering_balloon_phase_cannot_accept_a_previous_balloon_result():
    worker = StubWorker()
    tracker = ColorTracker()
    pipeline = FlightPipeline(worker, tracker, Mission("orange"))
    tracker.tracks = [Track(1, "green_balloon", [148., 108., 172., 132.], 0., 0.)]
    pipeline.configure("goal", "orange", 1.)
    assert tracker.tracks == []
    pipeline.configure("balloon", "orange", 2.)
    worker.results = [search_result(.5)]
    pipeline.step(rgb(), 2.1)
    assert tracker.tracks == []
    worker.results = [search_result(2.1)]
    pipeline.step(rgb(), 2.2)
    assert len(tracker.tracks) == 1


@pytest.mark.parametrize("distance", [float("nan"), float("inf"), -float("inf"), 0, -1, True])
def test_distance_must_be_a_positive_finite_measurement(distance):
    data = payload()
    data["distance_m"] = distance
    with pytest.raises(ValueError):
        validate_capture(data, "balloon", "orange")


@pytest.mark.parametrize("label", ["red_balloon", "orange_square", "background", "", "../green"])
def test_balloon_capture_rejects_non_balloon_or_unknown_label(label):
    data = payload()
    data["target_label"] = label
    with pytest.raises(ValueError):
        validate_capture(data, "balloon", "orange")


@pytest.mark.parametrize("count", [-1, 21, 1.2, True, float("nan")])
def test_expected_count_must_be_a_nonnegative_integer(count):
    data = payload()
    data["expected_count"] = count
    with pytest.raises(ValueError):
        validate_capture(data, "balloon", "orange")


def test_goal_capture_records_requested_color_and_exact_measured_distance():
    data = payload()
    data["target_label"] = "yellow_square"
    metadata = validate_capture(data, "goal", "yellow")
    assert metadata["distance_m"] == 1.2345
    assert metadata["target_label"] == "yellow_square"
    assert metadata.get("training_use", False) is False


def test_goal_capture_cannot_silently_label_the_other_goal_as_requested():
    data = payload()
    data["target_label"] = "orange_square"
    with pytest.raises(ValueError):
        validate_capture(data, "goal", "yellow")


def test_empty_scene_can_be_labelled_for_clutter_false_positive_review():
    data = payload()
    data["expected_count"] = 0
    assert validate_capture(data, "balloon", "orange")["expected_count"] == 0


def test_five_second_window_does_not_include_the_deadline_frame():
    buffer = CaptureBuffer()
    buffer.start(valid_metadata(), 10.)
    image = rgb()
    for when in (10., 10.05, 12.5, 14.999):
        assert buffer.append(image, captured_row(when), when)
    assert not buffer.append(image, captured_row(15.), 15.)
    assert len(buffer.frames) == 4
    status = buffer.status(now=15.)
    assert status["remaining_s"] == 0
    assert status["state"] != "complete"  # Export success has not happened yet.


def test_pre_button_inflight_frame_is_skipped_without_failing_the_take():
    buffer = CaptureBuffer()
    buffer.start(valid_metadata(), 10.)
    assert not buffer.append(rgb(), captured_row(9.99), 9.99)
    assert buffer.status(now=10.)["state"] == "capturing"
    assert buffer.append(rgb(), captured_row(10.05), 10.05)
    assert len(buffer.frames) == 1


def test_captured_pixels_and_telemetry_are_detached_from_live_mutations():
    buffer = CaptureBuffer()
    metadata = valid_metadata()
    buffer.start(metadata, 0.)
    image, row = rgb(), captured_row()
    expected_image, expected_row = image.copy(), deepcopy(row)
    expected_row["offset_s"] = 0.
    buffer.append(image, row, 0.)
    image.fill(255)
    row["tracks"][0]["box"][0] = 300.
    row["message"]["target"]["label"] = "purple_balloon"
    metadata["distance_m"] = 999.
    saved_image, saved_row = buffer.frames[0]
    assert np.array_equal(saved_image, expected_image)
    assert saved_row == expected_row
    assert buffer.metadata["distance_m"] == 1.2345


def test_duplicate_timestamp_fails_a_take_instead_of_duplicating_video_evidence():
    buffer = CaptureBuffer()
    buffer.start(valid_metadata(), 0.)
    assert buffer.append(rgb(), captured_row(.1), .1)
    assert not buffer.append(rgb(), captured_row(.1), .1)
    assert buffer.status(now=.1)["state"] == "failed"
    assert len(buffer.frames) == 1


@pytest.mark.parametrize("limit", ["frames", "bytes"])
def test_memory_limit_aborts_instead_of_reporting_a_complete_capture(limit):
    kwargs = dict(max_frames=1) if limit == "frames" else dict(max_bytes=rgb().nbytes)
    buffer = CaptureBuffer(**kwargs)
    buffer.start(valid_metadata(), 0.)
    buffer.append(rgb(), captured_row(0.), 0.)
    try:
        buffer.append(rgb(), captured_row(.05), .05)
    except (ValueError, RuntimeError, MemoryError):
        pass
    state = buffer.status(now=.1)
    assert state["state"] == "failed"
    assert state["error"]
    assert len(buffer.frames) <= 1


def finished_buffer():
    buffer = CaptureBuffer()
    buffer.start(valid_metadata(), 0.)
    for when in (0., .05, .1):
        buffer.append(rgb(), captured_row(when), when)
    buffer.append(rgb(), captured_row(5.), 5.)
    return buffer


def test_export_preserves_exact_labels_raw_pixels_and_recorded_boxes(tmp_path, monkeypatch):
    buffer = finished_buffer()
    # Rendering must use the captured telemetry rather than a second inference pass.
    def forbid_inference(*args, **kwargs):
        raise AssertionError("Export reran the flight pipeline")
    monkeypatch.setattr(FlightPipeline, "step", forbid_inference)
    identities = dict(scope="synthetic test", files={"model.tflite": "frozen-sha256"})
    summary = export_capture(buffer, tmp_path, identities)
    folder = tmp_path / buffer.status()["id"]
    assert (folder / "summary.json").is_file()
    scene = json.loads((folder / "scene.json").read_text())
    assert scene["distance_m"] == 1.2345
    assert scene["target_label"] == "green_balloon"
    rows = [json.loads(line) for line in (folder / "frames.jsonl").read_text().splitlines()]
    assert len(rows) == 3
    assert rows[0]["message"]["target"]["label"] == "green_balloon"
    assert rows[0]["tracks"][0]["box"] == [148., 108., 172., 132.]
    raw_paths = sorted((folder / "raw").glob("*.png"))
    assert len(raw_paths) == 3
    saved = cv2.cvtColor(cv2.imread(str(raw_paths[0])), cv2.COLOR_BGR2RGB)
    assert np.array_equal(saved, rgb())
    assert len(list((folder / "masks").glob("*.png"))) >= 3
    overlays = sorted((folder / "overlay").glob("*.png"))
    assert len(overlays) == 3
    overlay = cv2.cvtColor(cv2.imread(str(overlays[0])), cv2.COLOR_BGR2RGB)
    assert not np.array_equal(overlay, saved)
    assert summary.get("accuracy") is None


def test_export_failure_never_leaves_a_successful_summary(tmp_path, monkeypatch):
    buffer = finished_buffer()

    def fail_write(*args, **kwargs):
        raise OSError("Disk disconnected")

    monkeypatch.setattr(cv2, "imwrite", fail_write)
    with pytest.raises((OSError, RuntimeError)):
        export_capture(buffer, tmp_path, {})
    folder = tmp_path / buffer.status()["id"]
    assert not (folder / "summary.json").exists()
    assert buffer.status()["state"] != "complete"


def test_failed_png_encoder_return_value_is_an_export_failure(tmp_path, monkeypatch):
    buffer = finished_buffer()
    monkeypatch.setattr(cv2, "imwrite", lambda *args, **kwargs: False)
    with pytest.raises((OSError, RuntimeError)):
        export_capture(buffer, tmp_path, {})
    assert buffer.status()["state"] == "failed"
    assert not (tmp_path / buffer.status()["id"] / "summary.json").exists()


class NoCameraSource:
    """Only used in lifecycle tests; construction never opens any hardware."""

    closed = False

    def read(self):
        raise RuntimeError("Synthetic source disconnected")

    def close(self):
        self.closed = True


def synthetic_lab(root):
    pipeline = FlightPipeline(StubWorker(), ColorTracker(), Mission("orange"))
    lab = lab_web.Lab(NoCameraSource(), pipeline, root, {"source": "replay"})
    lab.ready = True
    lab.last_frame_at = time.monotonic()
    return lab


@pytest.fixture
def lab_server(tmp_path):
    lab = synthetic_lab(tmp_path)
    server = ThreadingHTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
    server.RequestHandlerClass = lab_web.make_handler(lab, server.server_port)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield lab, server.server_port
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def request(port, method, path, body=None, headers=None):
    connection = HTTPConnection("127.0.0.1", port, timeout=5)
    base_headers = {"Host": f"127.0.0.1:{port}"}
    if method == "POST":
        base_headers.update({"X-Lab-Request": "1", "Content-Type": "application/json"})
    base_headers.update(headers or {})
    if isinstance(body, (dict, list, bool)) or body is None and method == "POST":
        body = json.dumps(body)
    try:
        connection.request(method, path, body=body, headers=base_headers)
        response = connection.getresponse()
        return response.status, response.read(), dict(response.getheaders())
    finally:
        connection.close()


def test_http_state_reports_replay_source_without_opening_a_camera(lab_server):
    lab, port = lab_server
    status, body, headers = request(port, "GET", "/api/state")
    state = json.loads(body)
    assert status == 200 and state["ready"]
    assert state["source"] == "replay"
    assert state["capture"]["state"] == "idle"
    assert headers["Cache-Control"] == "no-store"
    assert lab.source.closed is False


@pytest.mark.parametrize("body", [[], True, "null", '"settings"', "not-json", b"\xff"])
def test_http_rejects_non_object_or_invalid_json(lab_server, body):
    lab, port = lab_server
    status, response, _ = request(port, "POST", "/api/capture", body)
    assert status == 400
    assert json.loads(response)["error"]
    assert lab.capture.state == "idle"


@pytest.mark.parametrize("headers", [
    {"Host": "attacker.invalid"},
    {"Host": "127.0.0.1.attacker.invalid:8770"},
    {"Origin": "https://attacker.invalid"},
    {"Origin": "null"},
    {"X-Lab-Request": ""},
    {"X-Lab-Request": "true"},
])
def test_http_rejects_cross_origin_and_missing_custom_header(lab_server, headers):
    lab, port = lab_server
    status, _, _ = request(port, "POST", "/api/capture", payload(), headers)
    assert status == 403
    assert lab.capture.state == "idle"


def test_http_get_also_rejects_untrusted_host(lab_server):
    _, port = lab_server
    status, _, _ = request(port, "GET", "/api/state", headers={"Host": "attacker.invalid"})
    assert status == 403


@pytest.mark.parametrize("body,headers", [
    ("x" * 4097, {}),
    ("{}", {"Content-Length": "-1"}),
    ("{}", {"Content-Length": "not-an-integer"}),
    ("", {"Content-Length": "0"}),
])
def test_http_rejects_oversized_or_malformed_body_length(lab_server, body, headers):
    lab, port = lab_server
    status, _, _ = request(port, "POST", "/api/capture", body, headers)
    assert status == 400
    assert lab.capture.state == "idle"


@pytest.mark.parametrize("distance", ["two", None, -1, float("nan"), float("inf")])
def test_http_rejects_malformed_distance_without_starting_capture(lab_server, distance):
    lab, port = lab_server
    data = payload()
    data["distance_m"] = distance
    status, _, _ = request(port, "POST", "/api/capture", data)
    assert status == 400
    assert lab.capture.state == "idle"


@pytest.mark.parametrize("busy_state", ["capturing", "exporting"])
def test_http_capture_and_config_return_conflict_while_busy(lab_server, busy_state):
    lab, port = lab_server
    lab.capture.start(valid_metadata(), time.monotonic())
    lab.capture.state = busy_state
    for path, settings in (("/api/capture", payload()),
                           ("/api/config", {"phase": "goal", "goal_color": "yellow"})):
        status, body, _ = request(port, "POST", path, settings)
        assert status == 409
        assert json.loads(body)["error"]
    assert lab.pending_config is None


def test_http_capture_and_config_require_pipeline_ready_after_reconfiguration(lab_server):
    lab, port = lab_server
    status, _, _ = request(port, "POST", "/api/config", {"phase": "goal", "goal_color": "yellow"})
    assert status == 200
    assert lab.pending_config == ("goal", "yellow")
    assert lab.ready is False
    assert request(port, "POST", "/api/capture", payload())[0] == 409
    assert lab.capture.state == "idle"


def test_http_capture_preserves_the_operator_distance_and_source(lab_server):
    lab, port = lab_server
    status, body, _ = request(port, "POST", "/api/capture", payload(),
                              headers={"Origin": f"http://127.0.0.1:{port}"})
    assert status == 202 and json.loads(body)["state"] == "capturing"
    assert lab.capture.metadata["distance_m"] == 1.2345
    assert lab.capture.metadata["source"] == "replay"
    assert lab.capture.metadata["training_use"] is False


def test_http_stale_camera_cannot_start_a_take(lab_server):
    lab, port = lab_server
    lab.last_frame_at = time.monotonic() - 10
    status, body, _ = request(port, "GET", "/api/state")
    assert status == 200 and not json.loads(body)["ready"]
    assert request(port, "POST", "/api/capture", payload())[0] == 409


SESSION = "20261102T120000-abcdef01"


@pytest.mark.parametrize("suffix", ["view", "summary.json", "download.zip", "raw/0000.png"])
def test_http_incomplete_session_has_no_download_or_review_surface(lab_server, suffix):
    lab, port = lab_server
    folder = lab.root / SESSION
    folder.mkdir()
    (folder / "download.zip").write_bytes(b"partial data")
    assert request(port, "GET", f"/sessions/{SESSION}/{suffix}")[0] == 404


@pytest.mark.parametrize("path", [
    "/sessions/../summary.json",
    f"/sessions/{SESSION}/../outside.json",
    f"/sessions/{SESSION}/%2e%2e/outside.json",
    f"/sessions/{SESSION}/raw/%2e%2e/scene.json",
    f"/sessions/{SESSION}/raw/0000.png/extra",
    "/sessions/not-a-session-id/view",
])
def test_http_session_paths_cannot_traverse_or_serve_arbitrary_files(lab_server, path):
    lab, port = lab_server
    folder = lab.root / SESSION
    folder.mkdir()
    (folder / "summary.json").write_text('{"complete":true}')
    (lab.root / "outside.json").write_text('{"secret":"not a capture"}')
    assert request(port, "GET", path)[0] == 404


def test_http_exported_take_review_uses_only_recorded_session_paths(lab_server):
    lab, port = lab_server
    buffer = finished_buffer()
    summary = export_capture(buffer, lab.root, {"source": "replay"})
    ident = summary["id"]
    status, body, headers = request(port, "GET", f"/sessions/{ident}/view")
    assert status == 200 and headers["Content-Type"].startswith("text/html")
    assert f"/sessions/{ident}/".encode() in body
    assert b"rows[index].offset_s" in body
    assert b"textContent" in body
    status, body, headers = request(port, "GET", f"/sessions/{ident}/raw/0000.png")
    assert status == 200 and headers["Content-Type"] == "image/png"
    assert body.startswith(b"\x89PNG")
    status, body, headers = request(port, "GET", f"/sessions/{ident}/download.zip")
    assert status == 200 and body.startswith(b"PK")
    assert "attachment" in headers["Content-Disposition"]


def test_source_failure_aborts_capture_and_releases_owned_resources(tmp_path):
    lab = synthetic_lab(tmp_path)
    lab.capture.start(valid_metadata(), time.monotonic())
    lab.loop()
    assert lab.capture.state == "failed"
    assert "Synthetic source disconnected" in lab.error
    assert lab.ready is False
    assert lab.source.closed
    assert lab.pipeline.worker.closed
    assert not list(tmp_path.glob("*/summary.json"))


def test_stalled_window_cannot_be_called_complete_even_when_export_finishes(tmp_path):
    buffer = CaptureBuffer()
    buffer.start(valid_metadata(), 10.)
    assert buffer.append(rgb(), captured_row(10.05), 10.05)
    assert buffer.append(rgb(), captured_row(10.10), 10.10)
    assert not buffer.append(rgb(), captured_row(30.), 30.)
    summary = export_capture(buffer, tmp_path, {"source": "replay"})
    assert summary["frames"] == 2
    assert summary["requested_duration_s"] == 5.
    assert summary["sample_span_s"] == pytest.approx(.05)
    assert summary["loop_fps"] == pytest.approx(20.)
    assert summary["export_complete"] is True
    assert summary["complete"] is False
    assert summary["termination_delay_ms"] == pytest.approx(15000.)
    assert summary["max_window_gap_ms"] >= 15000.
    assert summary["capture_error"]
    assert buffer.state == "failed"
    assert buffer.error
    folder = tmp_path / summary["id"]
    assert (folder / "download.zip").is_file()
    assert len(list((folder / "raw").glob("*.png"))) == 2
    saved_summary = json.loads((folder / "summary.json").read_text())
    assert saved_summary["complete"] is False
    assert saved_summary["export_complete"] is True


def window_buffer(times, finished=5.):
    buffer = CaptureBuffer()
    buffer.start(valid_metadata(), 0.)
    for when in times:
        assert buffer.append(rgb(), captured_row(when), when)
    assert not buffer.append(rgb(), captured_row(finished), finished)
    return buffer


def test_full_window_continuity_is_separate_from_accuracy_or_flight_approval(tmp_path):
    buffer = window_buffer([i / 5 for i in range(25)])
    summary = export_capture(buffer, tmp_path, {"source": "replay"})
    assert summary["complete"] is True
    assert summary["export_complete"] is True
    assert buffer.state == "complete"
    assert summary["start_gap_ms"] == 0.
    assert summary["end_gap_ms"] == pytest.approx(200.)
    assert summary["termination_delay_ms"] == 0.
    assert summary["max_window_gap_ms"] <= 250.
    assert summary.get("accuracy") is None
    assert "not a flight acceptance threshold" in summary["continuity_note"]


@pytest.mark.parametrize("times,finished,metric", [
    ([i / 5 for i in range(2, 25)], 5., "start_gap_ms"),
    ([i / 5 for i in range(23)], 5., "end_gap_ms"),
    ([i / 5 for i in range(25) if i != 12], 5., "max_gap_ms"),
    ([i / 5 for i in range(25)], 5.5, "termination_delay_ms"),
])
def test_capture_boundary_and_internal_gaps_are_all_in_integrity_result(
        tmp_path, times, finished, metric):
    buffer = window_buffer(times, finished)
    summary = export_capture(buffer, tmp_path, {"source": "replay"})
    assert summary[metric] > 250.
    assert summary["complete"] is False
    assert summary["export_complete"] is True
    assert buffer.state == "failed"
    assert (tmp_path / summary["id"] / "summary.json").is_file()


def test_existing_partial_directory_is_an_export_failure_without_overwrite(tmp_path):
    buffer = finished_buffer()
    pending = tmp_path / ("." + buffer.id + ".partial")
    pending.mkdir()
    old_evidence = pending / "keep-existing.txt"
    old_evidence.write_text("Existing interrupted export")
    with pytest.raises(FileExistsError):
        export_capture(buffer, tmp_path, {"source": "replay"})
    assert buffer.state == "failed"
    assert buffer.error.startswith("Export failed:")
    assert old_evidence.read_text() == "Existing interrupted export"
    assert not (tmp_path / buffer.id).exists()


@pytest.mark.parametrize("path,settings", [
    ("/api/capture", payload()),
    ("/api/config", {"phase": "goal", "goal_color": "yellow"}),
])
def test_http_active_artifact_transfer_blocks_new_configuration_or_capture(
        lab_server, path, settings):
    lab, port = lab_server
    lab.active_transfers = 1
    status, body, _ = request(port, "POST", path, settings)
    assert status == 409
    assert "playback or downloads" in json.loads(body)["error"]
    assert lab.capture.state == "idle"
    assert lab.pending_config is None


@pytest.mark.parametrize("busy_state", ["capturing", "exporting"])
def test_http_playback_downloads_cannot_compete_with_a_busy_take(lab_server, busy_state):
    lab, port = lab_server
    folder = lab.root / SESSION
    folder.mkdir()
    (folder / "summary.json").write_text('{"export_complete":true,"complete":true}')
    (folder / "download.zip").write_bytes(b"PKexisting take")
    lab.capture.state = busy_state
    for suffix in ("view", "summary.json", "download.zip", "raw/0000.png"):
        status, body, _ = request(port, "GET", f"/sessions/{SESSION}/{suffix}")
        assert status == 409
        assert "pause during capture" in json.loads(body)["error"]
        assert lab.active_transfers == 0


def test_http_transfer_counter_covers_the_actual_stream_and_always_releases(
        lab_server, monkeypatch):
    lab, port = lab_server
    folder = lab.root / SESSION
    folder.mkdir()
    (folder / "summary.json").write_text('{"export_complete":true,"complete":true}')
    (folder / "download.zip").write_bytes(b"PKexisting take")
    original_copy = lab_web.shutil.copyfileobj
    original_finish = lab_web.BaseHTTPRequestHandler.finish
    observed = []
    finished = threading.Event()

    def check_transfer(source, destination, length):
        observed.append(lab.active_transfers)
        with pytest.raises(RuntimeError, match="playback or downloads"):
            lab.start_capture(payload())
        original_copy(source, destination, length)

    def finish_request(handler):
        try:
            original_finish(handler)
        finally:
            finished.set()

    monkeypatch.setattr(lab_web.shutil, "copyfileobj", check_transfer)
    monkeypatch.setattr(lab_web.BaseHTTPRequestHandler, "finish", finish_request)
    status, body, _ = request(port, "GET", f"/sessions/{SESSION}/download.zip")
    assert status == 200 and body == b"PKexisting take"
    assert observed == [1]
    assert finished.wait(timeout=2)
    with lab.lock:
        assert lab.active_transfers == 0
    finished.clear()
    assert request(port, "GET", f"/sessions/{SESSION}/missing.json")[0] == 404
    assert finished.wait(timeout=2)
    with lab.lock:
        assert lab.active_transfers == 0


def test_stalled_capture_watchdog_fails_take_and_closes_resources_on_loop_exit(tmp_path):
    lab = synthetic_lab(tmp_path)
    lab.capture.start(valid_metadata(), 10.)
    lab.last_frame_at = 10.1
    lab.capture.append(rgb(), captured_row(10.1), 10.1)
    lab.check_health(now=11.2)
    assert lab.capture.state == "failed"
    assert "stream stalled" in lab.capture.error
    assert lab.error == lab.capture.error
    assert lab.ready is False and lab.running is False
    lab.loop()
    assert lab.source.closed and lab.pipeline.worker.closed
    assert "stream stalled" in lab.capture.error
    assert not list(tmp_path.glob("*/summary.json"))


def test_http_state_runs_the_watchdog_without_waiting_for_another_frame(lab_server):
    lab, port = lab_server
    now = time.monotonic()
    lab.capture.start(valid_metadata(), now - 2.)
    lab.last_frame_at = now - 2.
    status, body, _ = request(port, "GET", "/api/state")
    state = json.loads(body)
    assert status == 200
    assert state["capture"]["state"] == "failed"
    assert state["ready"] is False
    assert "stream stalled" in state["error"]
    assert lab.running is False


def test_http_server_checks_health_without_a_browser_poll(tmp_path):
    lab = synthetic_lab(tmp_path)
    now = time.monotonic()
    lab.capture.start(valid_metadata(), now - 2.)
    lab.last_frame_at = now - 2.
    server = lab_web.LabHTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
    server.lab = lab
    try:
        server.service_actions()
    finally:
        server.server_close()
    assert lab.capture.state == "failed"
    assert lab.ready is False and lab.running is False


def test_clean_camera_loop_shutdown_marks_active_take_interrupted(tmp_path):
    lab = synthetic_lab(tmp_path)
    lab.capture.start(valid_metadata(), time.monotonic())
    lab.running = False
    lab.loop()
    assert lab.capture.state == "failed"
    assert "stopped before the take finished" in lab.capture.error
    assert lab.source.closed and lab.pipeline.worker.closed


@pytest.mark.parametrize('color', ['orange', 'yellow'])
@pytest.mark.parametrize('shape', ['circle', 'square', 'triangle', 'goal'])
def test_capture_accepts_each_shape_and_colour_requests(color, shape):
    data = dict(payload(), target_label=f'{color}_{shape}')
    metadata = validate_capture(data, 'goal', color)
    assert metadata['target_label'] == f'{color}_{shape}'
    with pytest.raises(ValueError, match='colour'):
        validate_capture(data, 'goal', 'orange' if color == 'yellow' else 'yellow')


@pytest.mark.parametrize('requested', ['yellow_goal', 'yellow_square'])
def test_capture_requested_coverage_counts_any_shape_of_colour(tmp_path, requested):
    buffer = finished_buffer()
    buffer.metadata.update(target_label=requested, phase='goal', goal_color='yellow')
    for i, (_, row) in enumerate(buffer.frames):
        row['phase_before'] = 'goal'
        row['tracks'][0]['label'] = ['yellow_circle', 'yellow_triangle', 'orange_square'][i]
    summary = export_capture(buffer, tmp_path, {})
    assert summary['frames_with_requested_target'] == 2
    assert summary['requested_target_coverage'] == pytest.approx(2/3)


def test_goal_mask_colours_follow_hsv_order():
    from dtr.flight_lab import diagnostic_images
    image = np.zeros((240, 320, 3), np.uint8)
    image[40:80, 40:80] = (240, 110, 20)
    image[40:80, 160:200] = (230, 220, 30)
    mask, _, _ = diagnostic_images(image, {
        'phase_before': 'goal', 'tracks': [], 'message': {'cmd': 'SEARCH'}})
    assert tuple(mask[60, 60]) == (245, 145, 55)
    assert tuple(mask[60, 180]) == (235, 215, 65)


def test_reconfiguring_keeps_custom_target_sizes_and_config():
    from dtr.flight import MissionConfig
    from dtr.flight_pipeline import FlightPipeline
    sizes = deepcopy(Mission("orange").target_sizes)
    sizes["orange_circle"]["width_m"] = sizes["orange_circle"]["height_m"] = 2.0
    config = MissionConfig(enter_size=.3)
    pipeline = FlightPipeline(None, ColorTracker(), Mission("orange", config=config, target_sizes=sizes))
    pipeline.configure("goal", "yellow", 0.)
    assert pipeline.mission.target_sizes["orange_circle"]["width_m"] == 2.0
    assert pipeline.mission.config.enter_size == .3 and pipeline.mission.goal_color == "yellow"
