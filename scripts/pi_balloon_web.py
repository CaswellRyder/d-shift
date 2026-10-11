"""Browser viewer for the Pi balloon pipeline, for bench tests with real objects.

Runs the bundle's student on the live camera and serves each frame with its
detections to a browser page. The server binds to 127.0.0.1 only; open it from
the Mac through an SSH tunnel. Scene recording saves every raw frame and its
detections so a session can be scored later. No flight commands.
"""

import argparse
import base64
from collections import deque
from datetime import datetime
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

import cv2
import numpy as np
from PIL import Image

from dtr.data import read_json, sha256
from dtr.native_balloon_regions import NativeBalloonRegions
from dtr.research_search_schedule import SearchSchedule
from dtr.runtime import Predictor

try:
    from pi_balloon_live_bench import capture_rgb, frame_timing, validate_camera_config
    from pi_balloon_schedule_bench import scheduled_frame
    from pi_balloon_search_bench import check_predictions, process_frame
    from pi_red_blue_bench import health, verify_bundle
except ModuleNotFoundError:
    from scripts.pi_balloon_live_bench import capture_rgb, frame_timing, validate_camera_config
    from scripts.pi_balloon_schedule_bench import scheduled_frame
    from scripts.pi_balloon_search_bench import check_predictions, process_frame
    from scripts.pi_red_blue_bench import health, verify_bundle

PAGE = Path(__file__).with_name("pi_balloon_web.html")
SCENE = re.compile(r"[a-z0-9][a-z0-9_-]{0,47}")
MIN_FREE_BYTES = 300 * 1024 * 1024
KEEP = ("box", "crop_box", "label", "score", "scores", "accepted", "raw_accepted", "suppressed",
        "suppression_reason", "proposal_score", "area", "color_group")


# The Pi uses CLOCK_BOOTTIME like the live benches; macOS (replay only) lacks it.
CLOCK = getattr(time, "CLOCK_BOOTTIME", time.CLOCK_MONOTONIC)


def boottime_ns():
    return time.clock_gettime_ns(CLOCK)


class CameraSource:
    """Same camera mode as the live benches: 320x240 RGB888, full sensor, rotated 180."""

    def __init__(self, fps):
        from libcamera import Transform
        from picamera2 import Picamera2

        self.camera = Picamera2()
        try:
            self.camera.configure(self.camera.create_video_configuration(
                main={"size": (320, 240), "format": "RGB888"},
                sensor={"output_size": (2592, 1944), "bit_depth": 10},
                controls={"FrameRate": fps},
                buffer_count=2,
                queue=False,
                transform=Transform(hflip=True, vflip=True),
            ))
            validate_camera_config(self.camera.camera_configuration())
            self.camera.start()
        except BaseException:
            self.camera.close()
            raise
        time.sleep(1)

    def read(self):
        return capture_rgb(self.camera)

    def close(self):
        self.camera.close()


class ReplaySource:
    """Loops stored frames at a fixed rate, for trying the viewer off the Pi."""

    def __init__(self, paths, fps):
        if not paths:
            raise ValueError("No replay frames")
        self.frames = [np.asarray(Image.open(p).convert("RGB").resize((320, 240))) for p in paths]
        self.period, self.index, self.next = 1 / fps, 0, time.monotonic()

    def read(self):
        self.next += self.period
        time.sleep(max(0.0, self.next - time.monotonic()))
        rgb = self.frames[self.index % len(self.frames)]
        self.index += 1
        return rgb.copy(), {"SensorTimestamp": boottime_ns()}

    def close(self):
        pass


class Recorder:
    """Writes one scene: raw PNG frames, one JSON row per frame and scene.json."""

    def __init__(self, root):
        self.root, self.active, self.last = Path(root).expanduser(), None, None

    def start(self, scene, seconds, red, blue, notes, context):
        if self.active:
            raise ValueError("Already recording")
        if not SCENE.fullmatch(scene):
            raise ValueError("Scene name: lowercase letters, digits, - or _, up to 48 characters")
        if not 0 <= seconds <= 120:
            raise ValueError("Seconds must be 0..120 (0 saves one frame)")
        if not (0 <= red <= 20 and 0 <= blue <= 20):
            raise ValueError("Balloon counts must be 0..20")
        if len(notes) > 300:
            raise ValueError("Notes are limited to 300 characters")
        day = self.root / datetime.now().strftime("%Y%m%d")
        day.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(day).free < MIN_FREE_BYTES:
            raise ValueError("Less than 300 MB free on the Pi; not recording")
        take = 1 + sum(1 for p in day.glob(f"{scene}-*") if p.is_dir())
        folder = day / f"{scene}-{take:02d}"
        folder.mkdir()
        meta = dict(scene=scene, take=take, seconds=seconds, red_in_view=red, blue_in_view=blue,
                    notes=notes, started=datetime.now().isoformat(timespec="seconds"),
                    labels="Counts typed in the viewer before recording; not reviewed",
                    training_use=False, **context)
        (folder / "scene.json").write_text(json.dumps(meta, indent=2))
        self.active = dict(folder=folder, meta=meta, rows=(folder / "frames.jsonl").open("x"),
                           frames=0, accepted={"red_balloon": 0, "blue_balloon": 0},
                           deadline=time.monotonic() + seconds, single=seconds == 0)

    def write(self, rgb, row):
        state = self.active
        if not state:
            return
        name = f"{state['frames']:04d}.png"
        cv2.imwrite(str(state["folder"] / name), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_PNG_COMPRESSION, 1])
        state["rows"].write(json.dumps(dict(row, path=name), allow_nan=False) + "\n")
        state["frames"] += 1
        for d in row["detections"]:
            if d["accepted"] and d["label"] in state["accepted"]:
                state["accepted"][d["label"]] += 1
        if state["single"] or time.monotonic() >= state["deadline"]:
            self.stop()

    def stop(self):
        state, self.active = self.active, None
        if not state:
            return
        state["rows"].close()
        meta = dict(state["meta"], frames=state["frames"], accepted_detections=state["accepted"],
                    finished=datetime.now().isoformat(timespec="seconds"))
        (state["folder"] / "scene.json").write_text(json.dumps(meta, indent=2))
        self.last = dict(folder=str(state["folder"]), frames=state["frames"],
                         accepted=state["accepted"])

    def status(self):
        state = self.active
        if not state:
            return dict(active=False, last=self.last)
        return dict(active=True, folder=str(state["folder"]), frames=state["frames"],
                    remaining_s=max(0.0, state["deadline"] - time.monotonic()))


class Viewer:
    """Camera loop on one thread; HTTP handlers only read the latest published frame."""

    def __init__(self, source, infer, recorder, info):
        self.source, self.infer, self.recorder, self.info = source, infer, recorder, info
        self.policy = "full"
        self.lock = threading.Condition()
        self.latest = None
        self.sequence = 0
        self.watched = 0.0
        self.recent = deque(maxlen=30)
        self.health, self.health_at = {}, 0.0
        self.error = None
        self.running = True

    def loop(self):
        previous = None
        try:
            while self.running:
                start = boottime_ns()
                rgb, metadata = self.source.read()
                acquired = boottime_ns()
                with self.lock:
                    policy = self.policy
                observed = self.infer(rgb, policy)
                done = boottime_ns()
                timing = frame_timing(metadata, start, acquired, done, previous)
                if timing["timestamp_valid"]:
                    previous = timing["sensor_timestamp_ns"]
                row = dict(detections=[{k: d[k] for k in KEEP if k in d} for d in observed["detections"]],
                           mode=observed.get("mode", "mser_direct"), policy=policy,
                           **{k: observed[k] for k in ("processing_ms", "search_ms", "inference_ms",
                                                       "neural_calls")},
                           **timing)
                with self.lock:
                    self.recorder.write(rgb, row)
                    watched = time.monotonic() - self.watched < 3
                frame = None
                if watched:
                    ok, jpeg = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                            [cv2.IMWRITE_JPEG_QUALITY, 80])
                    frame = base64.b64encode(jpeg).decode("ascii") if ok else None
                if time.monotonic() - self.health_at > 10:
                    self.health, self.health_at = health(), time.monotonic()
                with self.lock:
                    self.recent.append((time.monotonic(), row))
                    self.sequence += 1
                    self.latest = dict(sequence=self.sequence, frame=frame, row=row)
                    self.lock.notify_all()
        except Exception as exc:  # Shown in the page; the camera is released below.
            with self.lock:
                self.error = f"{type(exc).__name__}: {exc}"
                self.lock.notify_all()
            raise
        finally:
            with self.lock:
                self.recorder.stop()
            self.source.close()

    def stats(self):
        rows = list(self.recent)
        if len(rows) < 2:
            return {}
        elapsed = rows[-1][0] - rows[0][0]

        def mean(key):
            values = [r[key] for _, r in rows if r.get(key) is not None]
            return float(np.mean(values)) if values else None

        return dict(fps=(len(rows) - 1) / elapsed if elapsed > 0 else None,
                    processing_ms=mean("processing_ms"), search_ms=mean("search_ms"),
                    inference_ms=mean("inference_ms"), neural_calls=mean("neural_calls"),
                    sensor_to_result_ms=mean("sensor_to_result_ms"), frames=len(rows))

    def state(self, after):
        with self.lock:
            self.watched = time.monotonic()
            self.lock.wait_for(lambda: self.error or (self.latest and self.latest["sequence"] > after),
                               timeout=2.0)
            latest = self.latest
            result = dict(error=self.error, policy=self.policy, stats=self.stats(),
                          health=self.health, recording=self.recorder.status())
        if latest:
            row = latest["row"]
            result.update(sequence=latest["sequence"], frame=latest["frame"], row=row,
                          age_ms=(boottime_ns() - row["result_boottime_ns"]) / 1e6)
        return result


def prepare(base, region_library, model_alias=None):
    """Bundle hashes and golden replay must pass before any camera opens."""
    root = Path(base).resolve()
    verify_bundle(root)
    native = NativeBalloonRegions(region_library, direct=True)
    receipt = read_json(native.path.with_suffix(".json"))
    if receipt["source_sha256"] != sha256(root / "dtr/native_balloon_regions.c"):
        raise ValueError("Native region library was built from different source")
    for key in ("DTR_RED_BLUE_COLOR_LOOKUP", "DTR_RED_BLUE_LOOKUP_LIBRARY"):
        os.environ.pop(key, None)
    cv2.setNumThreads(1)
    inputs = read_json(root / "inputs.json")
    alias = model_alias or inputs.get("default_model", "new-views-42")
    model = root / inputs["models"][alias]["path"]
    predictor = Predictor(model, allow_unvalidated=True)
    golden = read_json(root / "search-golden.json")[alias]
    deltas = []
    for mode in ("mser_direct", "mser_bright"):
        expected = golden["mser" if mode == "mser_direct" else mode]
        if len(expected) != len(inputs["frames"]) or not expected:
            raise ValueError("Incomplete goldens")
        for frame, truth in zip(inputs["frames"], expected, strict=True):
            rgb = np.asarray(Image.open(root / frame["path"]).convert("RGB"))
            deltas.append(check_predictions(process_frame(rgb, predictor, mode, native)["detections"],
                                            truth["detections"]))
    scheduler = SearchSchedule()

    def infer(rgb, policy):
        return scheduled_frame(rgb, predictor, native, scheduler, policy, boottime_ns)

    info = dict(model=alias, model_sha256=sha256(model), bundle_sha256=sha256(root / "bundle.json"),
                region_library_sha256=sha256(native.path), viewer_sha256=sha256(__file__),
                page_sha256=sha256(PAGE), threshold=predictor.metadata["threshold"],
                classes=predictor.metadata["classes"], max_golden_score_delta=max(deltas),
                golden_frames=len(inputs["frames"]))
    return infer, info, [root / f["path"] for f in inputs["frames"]]


def make_server(viewer, port):
    token = secrets.token_urlsafe(32)
    page = PAGE.read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(10)

        def log_message(self, *args):
            pass

        def send(self, status, body, content_type="application/json"):
            if not isinstance(body, bytes):
                body = json.dumps(body, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy",
                             "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
                             "script-src 'self' 'unsafe-inline'; object-src 'none'; frame-ancestors 'none'; "
                             "base-uri 'none'")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def trusted(self):
            # Through an SSH tunnel the browser still sends localhost:<port>.
            hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
            origin = self.headers.get("Origin")
            valid = (self.headers.get("Host") in hosts
                     and (origin is None or origin in {f"http://{h}" for h in hosts})
                     and self.headers.get("Sec-Fetch-Site") != "cross-site")
            if not valid:
                self.send(403, {"error": "Open the viewer at http://localhost:%d" % port})
            return valid

        def do_GET(self):
            if not self.trusted():
                return
            url = urlsplit(self.path)
            if url.path == "/":
                self.send(200, page, "text/html; charset=utf-8")
            elif url.path == "/api/config":
                self.send(200, dict(viewer.info, token=token))
            elif url.path == "/api/state":
                try:
                    after = int(parse_qs(url.query).get("after", ["0"])[0])
                except ValueError:
                    after = 0
                self.send(200, viewer.state(after))
            else:
                self.send(404, {"error": "Not found"})

        def do_POST(self):
            if not self.trusted():
                return
            if not secrets.compare_digest(self.headers.get("X-Token", ""), token):
                self.send(403, {"error": "Reload the page"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 <= length <= 4096:
                    raise ValueError("Request too large")
                body = json.loads(self.rfile.read(length) or b"{}")
                path = urlsplit(self.path).path
                with viewer.lock:
                    if path == "/api/record":
                        viewer.recorder.start(
                            str(body.get("scene", "")).strip().lower(), float(body.get("seconds", 10)),
                            int(body.get("red", 0)), int(body.get("blue", 0)), str(body.get("notes", "")),
                            dict(policy=viewer.policy, **viewer.info))
                    elif path == "/api/stop":
                        viewer.recorder.stop()
                    elif path == "/api/policy":
                        if body.get("policy") not in ("full", "periodic"):
                            raise ValueError("Policy must be full or periodic")
                        viewer.policy = body["policy"]
                    else:
                        self.send(404, {"error": "Not found"})
                        return
                    status = viewer.recorder.status()
                self.send(200, dict(ok=True, recording=status))
            except (ValueError, TypeError, OSError) as exc:
                self.send(400, {"error": str(exc)})

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Pi bundle directory")
    parser.add_argument("--region-library", required=True)
    parser.add_argument("--model", help="bundle model alias; default is the bundle's default_model")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--fps", type=float, default=10, help="requested camera rate")
    parser.add_argument("--sessions", default="~/lab-sessions", help="where recorded scenes go")
    parser.add_argument("--replay", nargs="*", help="loop these images instead of the camera "
                        "(no paths: the bundle's golden frames)")
    args = parser.parse_args()
    if not 1 <= args.fps <= 15:
        parser.error("fps must be 1..15")
    infer, info, golden_frames = prepare(args.base, args.region_library, args.model)
    print(f"Bundle and goldens OK: {info['model']} (max score delta {info['max_golden_score_delta']:.2e})",
          flush=True)
    if args.replay is None:
        source = CameraSource(args.fps)
        info["source"] = "camera"
    else:
        source = ReplaySource(args.replay or golden_frames, min(args.fps, 5))
        info["source"] = "replay"
    viewer = Viewer(source, infer, Recorder(args.sessions), info)
    server = make_server(viewer, args.port)
    camera = threading.Thread(target=viewer.loop, daemon=True)
    camera.start()
    print(f"Viewer on http://localhost:{args.port} (loopback only). From the Mac:\n"
          f"  ssh -N -L {args.port}:127.0.0.1:{args.port} pacman@vision.local\n"
          f"then open http://localhost:{args.port}. Ctrl+C stops and releases the camera.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        viewer.running = False
        server.server_close()
        camera.join(timeout=5)


if __name__ == "__main__":
    main()
