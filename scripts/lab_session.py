"""Laptop commands for the loopback Pi lab tunnel; Python standard library only.

Records rehearsal requests and bounded five-second takes. This client has no
ESP32, I2C, serial or actuator sender. A request acknowledgement is not motor action.
"""

import argparse
from datetime import datetime, timezone
import hashlib
from http.client import HTTPConnection, HTTPException
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import time
from urllib.parse import urlsplit
import uuid


SESSION_ID = re.compile(r"\d{8}T\d{6}-[0-9a-f]{8}")
SLUG = re.compile(r"[a-z0-9][a-z0-9_-]{0,47}")
ACTIONS = ("vacuum_intake", "vacuum_off", "vacuum_reverse", "capture_observed",
           "release_observed", "capture_failed", "reset_trial")
GOALS = tuple(f"{colour}_{shape}" for colour in ("orange", "yellow")
              for shape in ("circle", "square", "triangle"))
MAX_DOWNLOAD_BYTES = 512 * 1024 * 1024


class LabError(Exception):
    pass


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def finite_number(value, minimum, maximum, name):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not minimum <= value <= maximum):
        raise LabError(f"{name} must be from {minimum:g} to {maximum:g}")
    return value


def parse_url(url):
    try:
        parts = urlsplit(url)
        port = parts.port if parts.port is not None else 80
        host = parts.hostname
        loopback = host == "localhost" or ipaddress.ip_address(host).is_loopback
    except (ValueError, TypeError):
        raise LabError("Use a loopback HTTP URL, for example http://127.0.0.1:8770") from None
    if (parts.scheme != "http" or not loopback or parts.username is not None
            or parts.password is not None or parts.query or parts.fragment
            or parts.path not in ("", "/") or not 1 <= port <= 65535
            or any(char in url for char in ("\r", "\n", "\\"))):
        raise LabError("Use a loopback HTTP URL without credentials, path, query or fragment")
    return host, port


def strict_json(data):
    def invalid(value):
        raise ValueError("Non-finite JSON value: " + value)

    def floating(value):
        number = float(value)
        if not math.isfinite(number):
            invalid(value)
        return number

    result = json.loads(data, parse_constant=invalid, parse_float=floating)
    if not isinstance(result, dict):
        raise LabError("Lab response must be a JSON object")
    return result


def identity(state, require_full=False):
    if state.get("source") not in ("camera", "replay") or not isinstance(state.get("bundle"), str):
        raise LabError("The port is not an identifiable lab server")
    identities = state.get("identities")
    rehearsal = state.get("rehearsal") or {}
    session = rehearsal.get("server_session")
    if require_full and (not isinstance(identities, dict) or not identities.get("hashes")
                         or not isinstance(session, str) or not session):
        raise LabError("Server lacks source/model identity or server_session; update the lab kit")
    return dict(bundle=state["bundle"], source=state["source"],
                identities=identities, server_session=session)


def require_capabilities(state, *names):
    caps = state.get("capabilities", {})
    if not isinstance(caps, dict) or any(caps.get(name) is not True for name in names):
        raise LabError("This server lacks the requested capabilities; update the lab kit")


class LabClient:
    def __init__(self, url="http://127.0.0.1:8770", expected_bundle=None, timeout=10):
        self.host, self.port = parse_url(url)
        self.url = f"http://{'[' + self.host + ']' if ':' in self.host else self.host}:{self.port}"
        self.expected_bundle = expected_bundle
        self.timeout = finite_number(timeout, .1, 60, "Request timeout")
        self.identity = None

    def request(self, path, payload=None, timeout=None):
        if path not in ("/api/state", "/api/config", "/api/capture", "/api/rehearsal") and not re.fullmatch(
                r"/sessions/\d{8}T\d{6}-[0-9a-f]{8}/summary\.json", path):
            raise LabError("Invalid lab API path")
        conn = HTTPConnection(self.host, self.port, timeout=timeout or self.timeout)
        headers = {"X-Lab-Request": "1"}
        body = None
        if payload is not None:
            body = json.dumps(payload, allow_nan=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        try:
            # Each POST is sent once. Its outcome may be unknown after a connection failure.
            conn.request("POST" if body is not None else "GET", path, body, headers)
            response = conn.getresponse()
            data = response.read(2 * 1024 * 1024 + 1)
            if len(data) > 2 * 1024 * 1024:
                raise LabError("Lab JSON response exceeds 2 MiB")
            if response.status not in (200, 202):
                # HTTPConnection never follows redirects, including download redirects.
                raise LabError(f"Lab request failed with HTTP {response.status}")
            return strict_json(data)
        finally:
            conn.close()

    def state(self, timeout=None):
        state = self.request("/api/state", timeout=timeout)
        current = identity(state, require_full=self.identity is not None)
        if self.expected_bundle is not None and current["bundle"] != self.expected_bundle:
            raise LabError("Unexpected installed lab bundle; no further requests sent")
        if self.identity is not None and current != self.identity:
            raise LabError("Lab server, source or model identity changed; series stopped")
        return state

    def bind(self, state):
        self.identity = identity(state, require_full=True)

    def download(self, ident, output):
        if not isinstance(ident, str) or not SESSION_ID.fullmatch(ident):
            raise LabError("Invalid capture identifier")
        final = output / (ident + ".zip")
        partial = output / (ident + ".zip.partial")
        conn = HTTPConnection(self.host, self.port, timeout=self.timeout)
        copied, digest = 0, hashlib.sha256()
        try:
            conn.request("GET", f"/sessions/{ident}/download.zip")
            response = conn.getresponse()
            if response.status != 200:
                raise LabError(f"Capture download failed with HTTP {response.status}")
            raw_length = response.getheader("Content-Length")
            try:
                length = int(raw_length)
            except (ValueError, TypeError):
                raise LabError("Capture download needs a valid Content-Length") from None
            if not 0 < length <= MAX_DOWNLOAD_BYTES:
                raise LabError("Capture download exceeds the bounded size or is empty")
            fd = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, "wb") as destination:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    copied += len(block)
                    if copied > length:
                        raise LabError("Capture download exceeds its stated length")
                    destination.write(block)
                    digest.update(block)
                destination.flush()
                os.fsync(destination.fileno())
            if copied != length:
                raise LabError("Capture download was interrupted; partial evidence retained")
            # Exclusive publication: a concurrent completed ZIP is never overwritten.
            os.link(partial, final)
            partial.unlink()
            return dict(path=final.name, bytes=copied, sha256=digest.hexdigest(), complete=True)
        finally:
            conn.close()


def save_manifest(output, manifest):
    temporary = output / (".manifest-" + uuid.uuid4().hex + ".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(output / "manifest.json")
    finally:
        temporary.unlink(missing_ok=True)


def capture_settings(phase, goal_color, target, distance_m, scene="lab-series", notes="",
                     purpose="development", session_group=None, expected_count=1):
    if phase not in ("balloon", "goal") or goal_color not in ("orange", "yellow"):
        raise LabError("Choose balloon/goal and orange/yellow goal colour")
    labels = ("green_balloon", "purple_balloon") if phase == "balloon" else (
        GOALS + ("orange_goal", "yellow_goal"))
    if target not in labels or (phase == "goal" and not target.startswith(goal_color + "_")):
        raise LabError("Target must match the phase and selected goal colour")
    finite_number(distance_m, .000001, 100, "Measured distance")
    if purpose not in ("development", "training"):
        raise LabError("Purpose must be development or training")
    if not isinstance(scene, str) or not SLUG.fullmatch(scene):
        raise LabError("Scene must be a lowercase slug of at most 48 characters")
    if not isinstance(notes, str) or len(notes) > 500:
        raise LabError("Notes are limited to 500 characters")
    if session_group is None:
        session_group = datetime.now(timezone.utc).strftime("lab-%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    if not isinstance(session_group, str) or not SLUG.fullmatch(session_group):
        raise LabError("Session group must be a lowercase slug of at most 48 characters")
    if isinstance(expected_count, bool) or not isinstance(expected_count, int) or not 0 <= expected_count <= 20:
        raise LabError("Expected count must be an integer from 0 to 20")
    return dict(distance_m=distance_m, target_label=target, expected_count=expected_count,
                scene=scene, notes=notes, data_purpose=purpose, session_group=session_group)


def event_payload(action, duration_s=None, note=""):
    if action not in ACTIONS or not isinstance(note, str) or len(note) > 500:
        raise LabError("Invalid rehearsal action or note")
    payload = dict(action=action, note=note)
    if duration_s is not None:
        if action not in ("vacuum_intake", "vacuum_reverse"):
            raise LabError("Duration only applies to intake/reverse requests")
        payload["duration_s"] = finite_number(duration_s, .1, 5, "Rehearsal duration")
    return payload


def capture_series(client, output, settings, *, phase="balloon", goal_color="yellow", takes=3,
                   timeout_s=120, poll_s=.25, clock=time.monotonic, sleep=time.sleep):
    if isinstance(takes, bool) or not isinstance(takes, int) or not 1 <= takes <= 100:
        raise LabError("Takes must be an integer from 1 to 100")
    finite_number(timeout_s, .1, 600, "Stage timeout")
    finite_number(poll_s, .05, 5, "Poll interval")
    output = Path(output)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    manifest = dict(schema="lab-capture-series-v1", status="starting", complete=False,
                    created_at=utc_now(), url=client.url, remote_identity=None, settings=settings,
                    phase=phase, goal_color=goal_color, requested_takes=takes,
                    source_scope="New bounded takes; gaps between capture/export/download are not recorded",
                    training_use=settings["data_purpose"] == "training", takes=[],
                    actuator_transport=False, controller_application="unavailable")
    save_manifest(output, manifest)

    def wait_for(predicate):
        deadline = clock() + timeout_s
        while True:
            remaining = deadline - clock()
            if remaining <= 0:
                raise LabError("Timed out waiting for the lab; no request was replayed")
            state = client.state(timeout=min(client.timeout, remaining))
            if state.get("error"):
                raise LabError("Lab reports an error; series stopped")
            if predicate(state):
                return state
            sleep(min(poll_s, max(0, deadline - clock())))

    try:
        state = client.state()
        require_capabilities(state, "capture_series", "data_purpose", "rehearsal")
        client.bind(state)
        manifest["remote_identity"] = client.identity
        if state.get("capture", {}).get("state") in ("capturing", "exporting"):
            raise LabError("Another take is running; wait before starting a series")
        manifest["status"] = "configuring"
        save_manifest(output, manifest)
        client.request("/api/config", dict(phase=phase, goal_color=goal_color))
        wait_for(lambda s: s.get("ready") is True and s.get("phase") == phase
                 and s.get("goal_color") == goal_color)
        for index in range(takes):
            current = client.state()
            if (current.get("phase") != phase or current.get("goal_color") != goal_color
                    or current.get("ready") is not True
                    or current.get("capture", {}).get("state") in ("capturing", "exporting")):
                raise LabError("Lab phase/readiness changed before the next take; series stopped")
            row = dict(index=index + 1, id=None, status="start_pending",
                       remote_capture_complete=False, local_download_complete=False)
            manifest["takes"].append(row)
            manifest["status"] = "recording"
            save_manifest(output, manifest)
            try:
                started = client.request("/api/capture", settings)
            except (OSError, HTTPException, ValueError, LabError):
                row["status"] = "start_outcome_unknown"
                raise
            ident = started.get("id")
            if not isinstance(ident, str) or not SESSION_ID.fullmatch(ident):
                row["status"] = "invalid_start_response"
                raise LabError("Capture start returned an invalid identifier")
            row.update(id=ident, status="capturing")
            save_manifest(output, manifest)

            def done(s):
                capture = s.get("capture", {})
                if capture.get("id") != ident:
                    raise LabError("Capture identity changed; another take may have replaced this take")
                if capture.get("state") == "failed":
                    raise LabError("Capture or export failed; incomplete take retained on Pi")
                return capture.get("state") == "complete"

            wait_for(done)
            summary = client.request(f"/sessions/{ident}/summary.json")
            if (summary.get("id") != ident or summary.get("complete") is not True
                    or summary.get("export_complete") is not True):
                raise LabError("Take lacks a matching complete recording and export")
            if summary.get("identities") != client.identity["identities"]:
                raise LabError("Take source/model identities differ from the pinned server")
            if (summary.get("data_purpose") != settings["data_purpose"]
                    or summary.get("session_group") != settings["session_group"]
                    or summary.get("training_use") is not (settings["data_purpose"] == "training")):
                raise LabError("Take data purpose/group differs from the requested collection")
            row.update(status="downloading", remote_capture_complete=True, summary=summary)
            manifest["status"] = "downloading"
            save_manifest(output, manifest)
            # Re-check the running session before and after the bounded download.
            client.state()
            row["download"] = client.download(ident, output)
            client.state()
            row.update(status="complete", local_download_complete=True)
            save_manifest(output, manifest)
            if index + 1 < takes:
                wait_for(lambda s: s.get("ready") is True
                         and s.get("capture", {}).get("state") not in ("capturing", "exporting"))
        manifest.update(status="complete", complete=True, finished_at=utc_now())
    except KeyboardInterrupt:
        manifest.update(status="interrupted", finished_at=utc_now(),
                        note="Pi capture/request may still be running; inspect status before another command")
        save_manifest(output, manifest)
        raise
    except (OSError, HTTPException, ValueError, LabError) as error:
        manifest.update(status="failed", finished_at=utc_now(), error=str(error))
        save_manifest(output, manifest)
        raise
    save_manifest(output, manifest)
    return manifest


def monitor(client, seconds=10, poll_s=.5, clock=time.monotonic, sleep=time.sleep):
    finite_number(seconds, .1, 600, "Monitor duration")
    finite_number(poll_s, .05, 5, "Poll interval")
    deadline = clock() + seconds
    first = client.state()
    require_capabilities(first, "rehearsal")
    client.bind(first)
    while clock() < deadline:
        state = client.state(timeout=min(client.timeout, max(.01, deadline - clock())))
        age = state.get("frame_age_ms")
        fresh = (state.get("ready") is True and type(age) in (int, float)
                 and math.isfinite(age) and 0 <= age <= 2000)
        rehearsal = state.get("rehearsal") or {}
        model = rehearsal.get("model_request") if fresh else None
        message = model if isinstance(model, dict) else None
        telemetry = state.get("flight_telemetry") if fresh else None
        telemetry = telemetry or {}
        outgoing = telemetry.get("outgoing") or {}
        print(json.dumps(dict(phase=state.get("phase"), ready=state.get("ready"),
                              frame_age_ms=age, observation_status="current" if fresh else "stale_or_unavailable",
                              capture=state.get("capture"), model_message=message or None,
                              navigation_request=outgoing.get("draft_packet"),
                              rehearsal=state.get("rehearsal"),
                              transmitted=False, applied=False), allow_nan=False), flush=True)
        sleep(min(poll_s, max(0, deadline - clock())))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8770")
    parser.add_argument("--expect-bundle")
    parser.add_argument("--request-timeout-s", type=float, default=10)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status", help="Show actual server state and installed identity")
    event = commands.add_parser("event", help="Record an operator request/observation; no motor sender")
    event.add_argument("action", choices=ACTIONS)
    event.add_argument("--duration-s", type=float)
    event.add_argument("--note", default="")
    watch = commands.add_parser("monitor", help="Print bounded live model/rehearsal telemetry")
    watch.add_argument("--seconds", type=float, default=10)
    watch.add_argument("--poll-s", type=float, default=.5)
    series = commands.add_parser("capture-series", help="Record/download bounded five-second takes")
    series.add_argument("--takes", type=int, default=3)
    series.add_argument("--phase", choices=("balloon", "goal"), default="balloon")
    series.add_argument("--goal-color", choices=("orange", "yellow"), default="yellow")
    series.add_argument("--target", required=True)
    series.add_argument("--distance-m", type=float, required=True)
    series.add_argument("--expected-count", type=int, default=1)
    series.add_argument("--scene", default="lab-series")
    series.add_argument("--notes", default="")
    series.add_argument("--purpose", choices=("development", "training"), default="development")
    series.add_argument("--session-group")
    series.add_argument("--output", type=Path, required=True, help="New private output directory")
    series.add_argument("--timeout-s", type=float, default=120, help="Per-stage wait limit, max 600 s")
    series.add_argument("--poll-s", type=float, default=.25)
    args = parser.parse_args(argv)
    try:
        client = LabClient(args.url, args.expect_bundle, args.request_timeout_s)
        if args.command == "status":
            print(json.dumps(client.state(), indent=2, allow_nan=False))
        elif args.command == "event":
            payload = event_payload(args.action, args.duration_s, args.note)
            state = client.state()
            require_capabilities(state, "rehearsal")
            try:
                result = client.request("/api/rehearsal", payload)
            except (OSError, HTTPException, ValueError):
                raise LabError("Rehearsal request outcome unknown; inspect status before retrying") from None
            print(json.dumps(dict(receipt=result, actuator_application="unavailable"), allow_nan=False))
        elif args.command == "monitor":
            monitor(client, args.seconds, args.poll_s)
        else:
            settings = capture_settings(args.phase, args.goal_color, args.target, args.distance_m,
                                        args.scene, args.notes, args.purpose, args.session_group,
                                        args.expected_count)
            result = capture_series(client, args.output, settings, phase=args.phase,
                                    goal_color=args.goal_color, takes=args.takes,
                                    timeout_s=args.timeout_s, poll_s=args.poll_s)
            print(json.dumps(dict(status=result["status"], takes=len(result["takes"]),
                                  manifest=str(args.output / "manifest.json"),
                                  applied=False, transmitted_to_controller=False)))
        return 0
    except KeyboardInterrupt:
        print("Interrupted. A Pi request may still be running; inspect status before retrying.")
        return 130
    except (LabError, OSError, HTTPException, ValueError, TypeError) as error:
        print("Lab session: " + str(error))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
