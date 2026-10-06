"""Loopback-only webcam debug UI. No camera access until browser user opts in.

Research utility, not a production web server. Never bind this to a public interface.
Frames exist only in memory; filesystem serving is restricted to named static UI assets.
"""

import argparse
import base64
import io
import json
import secrets
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from .runtime import Predictor
from .tracking import BoxTracker
from .vision import DEFAULT_DUPLICATE_POLICY, DEFAULT_LIMIT, color_mask, observe

ASSETS = Path(__file__).with_name("static")
MAX_BODY = 256 * 1024


class VisionService:
    def __init__(
        self,
        models,
        allow_unvalidated=False,
        goal_proposal_limit=12,
        proposal_profile="balloon_components",
    ):
        if proposal_profile not in ("v2", "balloon_components"):
            raise ValueError("Viewer proposal profile must be v2 or balloon_components")
        self.profile = proposal_profile
        if type(goal_proposal_limit) is not int or goal_proposal_limit not in (12, 24):
            raise ValueError("Goal proposal limit must be 12 or 24")
        self.limits = {"balloon": DEFAULT_LIMIT, "goal": goal_proposal_limit}
        self.predictors = {}
        for task, path in models.items():
            if Path(path).suffix == ".keras":
                from .teacher_runtime import TeacherPredictor

                self.predictors[task] = TeacherPredictor(path, allow_unvalidated)
            else:
                self.predictors[task] = Predictor(path, allow_unvalidated)
        for task, predictor in self.predictors.items():
            if predictor.metadata["task"] != task:
                raise ValueError(f"Wrong task in {task} model metadata")
        self.sessions = {}
        self.lock = threading.Lock()

    def config(self):
        return {
            "tasks": {
                task: {
                    "classes": p.metadata["classes"],
                    "synthetic": p.metadata["synthetic_training"],
                    "threshold": p.metadata["threshold"],
                    "deployment_approved": p.metadata["deployment_approved"],
                    "engine": p.metadata.get("kind", "int8_student"),
                    "model_sha256": p.metadata.get("sha256"),
                    "proposal_limit": self.limits[task],
                }
                for task, p in self.predictors.items()
            },
            "width": 320,
            "height": 240,
            "proposal_limit": DEFAULT_LIMIT,
            "proposal_profile": self.profile,
            "duplicate_policy": DEFAULT_DUPLICATE_POLICY,
        }

    def process(self, body, task, session, sequence, reset=False):
        if task not in self.predictors:
            raise ValueError("Unknown task")
        if not session or len(session) > 64 or not session.replace("-", "").isalnum():
            raise ValueError("Invalid session")
        if sequence < 0:
            raise ValueError("Invalid frame sequence")
        with Image.open(io.BytesIO(body)) as opened:
            if opened.size != (320, 240) or opened.format != "JPEG":
                raise ValueError("Expected a 320x240 JPEG frame")
            rgb = np.asarray(opened.convert("RGB"))
        with self.lock:
            now = time.monotonic()
            self.sessions = {k: v for k, v in self.sessions.items() if now - v["last"] < 60}
            key = (session, task)
            if key not in self.sessions:
                if len(self.sessions) >= 8:
                    raise ValueError("Too many active sessions; close extra tabs and wait a minute")
                self.sessions[key] = {"tracker": BoxTracker(), "last": now, "sequence": -1}
            state = self.sessions[key]
            if sequence <= state["sequence"]:
                raise ValueError("Stale or duplicate frame")
            if reset:
                state["tracker"] = BoxTracker()
            result = observe(
                rgb, self.predictors[task], limit=self.limits[task], profile=self.profile
            )
            result["observations"] = state["tracker"].update(result["observations"], now)
            state.update(last=now, sequence=sequence)
            ok, png = cv2.imencode(".png", color_mask(rgb, task, self.profile))
            if not ok:
                raise ValueError("Mask encoding failed")
            result.update(
                mask="data:image/png;base64," + base64.b64encode(png).decode("ascii"),
                sequence=sequence,
                task=task,
                synthetic_training=self.predictors[task].metadata["synthetic_training"],
                threshold=self.predictors[task].metadata["threshold"],
                tracking="greedy IoU; candidate identity only; not a flight tracker",
            )
            return result


def make_server(service, port=8765, public_origins=()):
    token = secrets.token_urlsafe(32)
    allowed_public = set()
    for origin in public_origins:
        parsed = urlsplit(origin)
        if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
                or parsed.path or parsed.query or parsed.fragment or "*" in origin
                or any(c.isspace() for c in origin)):
            raise ValueError("Public origin must be an exact HTTPS origin without a path")
        parsed.port  # Validate port syntax before opening a socket.
        allowed_public.add(origin)

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(10)

        def log_message(self, *args):
            pass  # Never log camera payloads or per-frame requests.

        def send(self, status, body, content_type="application/json"):
            if not isinstance(body, bytes):
                body = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Permissions-Policy", "camera=(self), microphone=()")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; img-src 'self' data: blob:; media-src 'self' blob:; object-src 'none'; frame-ancestors 'none'; base-uri 'none'",
            )
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def trusted(self):
            hosts = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            origins = {f"http://{h}" for h in hosts} | allowed_public
            hosts |= {urlsplit(o).netloc for o in allowed_public}
            origin = self.headers.get("Origin")
            valid = (
                self.headers.get("Host") in hosts
                and (origin is None or (origin in origins
                     and urlsplit(origin).netloc == self.headers.get("Host")))
                and self.headers.get("Sec-Fetch-Site") != "cross-site"
            )
            if not valid:
                self.send(403, {"error": "Configured same-origin requests only"})
            return valid

        def do_GET(self):
            if not self.trusted():
                return
            path = urlsplit(self.path).path
            if path == "/api/config":
                self.send(200, {**service.config(), "token": token})
                return
            assets = {
                "/": ("index.html", "text/html; charset=utf-8"),
                "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                "/frame-policy.js": ("frame-policy.js", "text/javascript; charset=utf-8"),
                "/goal-view.js": ("goal-view.js", "text/javascript; charset=utf-8"),
                "/style.css": ("style.css", "text/css; charset=utf-8"),
            }
            if path not in assets:
                self.send(404, {"error": "Not found"})
                return
            filename, mime = assets[path]
            self.send(200, (ASSETS / filename).read_bytes(), mime)

        def do_POST(self):
            if not self.trusted():
                return
            if not secrets.compare_digest(self.headers.get("X-DTR-Token", ""), token):
                self.send(403, {"error": "Reload the viewer to authorize local inference"})
                return
            url = urlsplit(self.path)
            if url.path != "/api/frame":
                self.send(404, {"error": "Not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_BODY or self.headers.get("Content-Type") != "image/jpeg":
                    raise ValueError("Expected JPEG body, at most 256 KiB")
                query = parse_qs(url.query)
                body = self.rfile.read(length)
                if len(body) != length:
                    raise ValueError("Incomplete frame")
                result = service.process(
                    body,
                    query["task"][0],
                    query["session"][0],
                    int(query["sequence"][0]),
                    reset=query.get("reset", ["0"])[0] == "1",
                )
                self.send(200, result)
            except (ValueError, KeyError, IndexError, UnidentifiedImageError, OSError) as exc:
                self.send(400, {"error": str(exc)})
            except Exception:
                self.send(500, {"error": "Inference failed; stop and check the model/runtime"})

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Local DTR camera viewer; no flight commands")
    parser.add_argument("--balloon-model", default="runs/balloon-proposals-20261005/teacher.keras")
    parser.add_argument("--goal-model", default="runs/goal-proposals-20261005/teacher.keras")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--allow-unvalidated", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--public-origin", action="append", default=[],
                        help="Exact HTTPS origin of a trusted private reverse proxy; repeatable")
    parser.add_argument(
        "--proposal-profile",
        choices=["v2", "balloon_components"],
        default="balloon_components",
        help="Use v2 only for baseline comparisons",
    )
    parser.add_argument(
        "--goal-proposal-limit",
        type=int,
        choices=[12, 24],
        default=12,
        help="24 is a higher-compute desktop experiment, not a Pi budget",
    )
    args = parser.parse_args(argv)
    models = {
        k: v
        for k, v in {"balloon": args.balloon_model, "goal": args.goal_model}.items()
        if Path(v).is_file()
    }
    if not models:
        parser.error(
            "No models found. Run from project root or provide --balloon-model / --goal-model"
        )
    service = VisionService(
        models, args.allow_unvalidated, args.goal_proposal_limit, args.proposal_profile
    )
    server = make_server(service, args.port, args.public_origin)
    url = f"http://127.0.0.1:{server.server_port}"
    print(
        f"DTR Vision: {url}\nClick Start camera in your browser. Ctrl+C stops the server.",
        flush=True,
    )
    print("Research viewer only. No recording, external inference, or flight commands.", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
