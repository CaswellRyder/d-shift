"""Flight-pipeline lab console. Loopback HTTP, five-second RAM capture, deferred export.

Start through an SSH tunnel. No serial, radio, motor, or flight-controller transport.
"""

import argparse
import base64
from collections import deque
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import shutil
import signal
import threading
import time
from urllib.parse import unquote, urlsplit

import cv2
import numpy as np

from dtr.flight import ColorTracker, Mission
from dtr.flight_lab import CaptureBuffer, diagnostic_images, export_capture, validate_capture
from dtr.flight_pipeline import FlightPipeline
from dtr.raw_capture import spool_capture
from dtr.lab_rehearsal import Rehearsal, validate_event
from dtr.target_geometry import load_target_sizes
from dtr.target_vector import VectorCreator
from dtr.flight_telemetry import add_mapping_arguments, mapping_from_args, mapping_identity_paths
from dtr.search_worker import DEFAULT_BALLOON_LIMIT, ModelSearch, SearchWorker

try:
    from scripts.pi_flight_loop import Camera, Replay
except ModuleNotFoundError:
    from pi_flight_loop import Camera, Replay

PAGE = Path(__file__).resolve().parents[1] / 'src/dtr/static/flight_lab.html'
BUSY = ('capturing', 'exporting')
SESSION_ID = re.compile(r'\d{8}T\d{6}-[0-9a-f]{8}')


def jpeg(rgb):
    ok, encoded = cv2.imencode('.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, 75])
    if not ok:
        raise OSError('Preview encoding failed')
    return base64.b64encode(encoded).decode('ascii')


class Lab:
    def __init__(self, source, pipeline, root, identities, capture_mode="legacy"):
        self.capture_mode = capture_mode
        self.spool_thread = None
        self.source, self.pipeline = source, pipeline
        self.root, self.identities = Path(root), identities
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.capture = CaptureBuffer(wall_clock_offset_s=identities.get('wall_clock_offset_s', 0.))
        self.active_transfers = 0
        self.running, self.ready, self.error = True, False, None
        self.phase, self.goal_color = 'balloon', pipeline.mission.goal_color
        self.pending_config, self.frame = None, None
        self.flight_telemetry = None
        self.last_row, self.last_frame_time = None, None
        self.rehearsal = Rehearsal(self._record_rehearsal_event)
        self.event_log_name = f'operator-events-{self.rehearsal.server_session}.jsonl'
        fd = os.open(self.root / self.event_log_name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(fd)
        self.sequence, self.last_frame_at, self.preview_at, self.watched_at = 0, 0., 0., 0.
        self.times, self.refresh = deque(maxlen=100), deque(maxlen=20)
        self.sessions = []
        for path in sorted(self.root.glob('*/summary.json'), reverse=True):
            try:
                result = json.loads(path.read_text())
                if result.get('export_complete', result.get('complete')) and SESSION_ID.fullmatch(path.parent.name):
                    self.sessions.append(result)
            except (OSError, ValueError):
                continue
        self.sessions = self.sessions[:100]

    def configure(self, payload):
        phase, colour = payload.get('phase'), payload.get('goal_color')
        if phase not in ('balloon', 'goal') or colour not in ('orange', 'yellow'):
            raise ValueError('Choose balloon/goal and orange/yellow goal colour')
        with self.lock:
            if self.capture.state in BUSY:
                raise RuntimeError('Wait until the current take is saved')
            if self.active_transfers:
                raise RuntimeError('Wait for take playback or downloads to finish before capture')
            if self.rehearsal.pending:
                raise RuntimeError('Wait for pending rehearsal events to reach the next frame')
            self.pending_config = (phase, colour)
            self.ready = False

    def start_capture(self, payload):
        with self.lock:
            if self.capture.state in BUSY:
                raise RuntimeError('Wait until the current take is saved')
            if self.active_transfers:
                raise RuntimeError('Wait for take playback or downloads to finish before capture')
            if not self.ready or self.error or time.monotonic() - self.last_frame_at > 2:
                raise RuntimeError('Camera and flight pipeline must be ready before capture')
            if shutil.disk_usage(self.root).free < 500 * 1024 * 1024:
                raise ValueError('At least 500 MB free space is required')
            metadata = validate_capture(payload, self.phase, self.goal_color)
            metadata['source'] = self.identities['source']
            metadata['worker_errors_before'] = self.pipeline.worker.errors
            metadata['worker_restarts_before'] = self.pipeline.worker.restarts
            metadata['rehearsal_server_session'] = self.rehearsal.server_session
            metadata['operator_event_log'] = self.event_log_name
            metadata['rehearsal_initial'] = self.rehearsal.snapshot(time.monotonic(),
                self.last_row, source=self.identities['source'], frame_sequence=self.sequence)
            ident = self.capture.start(metadata, time.monotonic())
            return dict(id=ident, state='capturing')

    def _record_rehearsal_event(self, event):
        # Small operator records are persisted by the request/frame owner. Image
        # encoding remains deferred; this measured path does not claim no I/O.
        previous = len(self.capture.events)
        self.capture.record_event(event)
        try:
            with (self.root / self.event_log_name).open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(event, allow_nan=False) + '\n')
                stream.flush()
        except OSError:
            del self.capture.events[previous:]
            raise

    def record_rehearsal(self, payload):
        settings = validate_event(payload)
        with self.lock:
            now = time.monotonic()
            if settings['action'] != 'vacuum_off':
                if self.capture.state == 'exporting':
                    raise RuntimeError('Wait until the current take is saved')
                if (not self.ready or self.error or self.pending_config is not None
                        or now - self.last_frame_at > 2):
                    raise RuntimeError('Camera and flight pipeline must be ready for rehearsal')
            event = self.rehearsal.receive(payload, now, frame_sequence=self.sequence,
                                           frame_time=self.last_frame_time)
            return dict(event, event_log=self.event_log_name)

    def _apply_rehearsal(self, now):
        def transition(phase, when):
            self.pipeline.configure(phase, self.goal_color, when)
            self.phase = phase
            self.times.clear()
            self.refresh.clear()
        return self.rehearsal.apply(now, transition)

    def telemetry(self):
        state = self.state(watch=False)
        result = state['capture'].get('result')
        if result is not None:
            state['capture']['result'] = {k: result.get(k) for k in
                ('id', 'bytes', 'sha256', 'frames', 'created_at', 'source',
                 'complete', 'export_complete', 'capture_error') if k in result}
        return {k: state[k] for k in ("source", "ready", "error", "frame_age_ms", "phase",
                "goal_color", "metrics", "flight_telemetry", "capture", "bundle")}

    def raw_captures(self):
        result = []
        for path in sorted(self.root.glob("*/raw-summary.json"), reverse=True):
            if not SESSION_ID.fullmatch(path.parent.name) or path.is_symlink():
                continue
            try:
                entry = json.loads(path.read_text())
                archive = path.parent / "capture.tar"
                if entry["id"] == path.parent.name and archive.is_file() and not archive.is_symlink():
                    result.append(entry)
            except (OSError, ValueError, KeyError):
                continue
        return result[:100]

    def state(self, watch=True):
        self.check_health()
        with self.lock:
            now = time.monotonic()
            if watch:
                self.watched_at = now
            stale = now - self.last_frame_at > 3
            gaps = np.diff(self.times)
            fps = float((len(self.times) - 1) / (self.times[-1] - self.times[0])) if len(self.times) > 1 else None
            refresh = np.diff(self.refresh)
            return dict(ready=self.ready and not stale and not self.error,
                        error=self.error, phase=self.phase, goal_color=self.goal_color,
                        source=self.identities['source'],
                        bundle=self.identities.get('bundle'),
                        identities=self.identities,
                        capabilities=dict(rehearsal=True, capture_series=True, data_purpose=True),
                        rehearsal=dict(self.rehearsal.snapshot(now, None if stale else self.last_row,
                                            source=self.identities['source'], frame_sequence=self.sequence),
                                       recent_events=list(self.rehearsal.recent),
                                       pending_events=len(self.rehearsal.pending), event_log=self.event_log_name),
                        flight_telemetry=self.flight_telemetry if not stale else None,
                        frame=None if not watch or self.capture.state in BUSY else self.frame,
                        frame_age_ms=(now - self.last_frame_at) * 1000 if self.last_frame_at else None,
                        metrics=dict(fps=fps, max_gap_ms=float(gaps.max() * 1000) if len(gaps) else None,
                                     search_refresh_ms=float(refresh.mean() * 1000) if len(refresh) else None,
                                     worker_errors=self.pipeline.worker.errors,
                                     worker_restarts=self.pipeline.worker.restarts),
                        capture=self.capture.status(now), sessions=[{k: s.get(k) for k in
                            ('id', 'scene', 'phase', 'distance_m', 'target_label', 'frames',
                             'created_at', 'summary_url', 'view_url', 'download_url',
                             'complete', 'export_complete', 'capture_error')}
                            for s in self.sessions])

    def check_health(self, now=None):
        now = time.monotonic() if now is None else now
        with self.lock:
            if (self.capture.state == 'capturing' and now - self.last_frame_at > 1
                    and now - self.capture.started > 1):
                self.error = 'Camera frame stream stalled; restart the viewer before another take'
                self.ready, self.running = False, False
                self.capture.finish(self.error, now=now)

    def loop(self):
        try:
            while self.running:
                with self.lock:
                    if self.pending_config is not None:
                        self.phase, self.goal_color = self.pending_config
                        self.pipeline.configure(self.phase, self.goal_color, time.monotonic())
                        self.pending_config = None
                        self.times.clear()
                        self.refresh.clear()
                rgb, now = self.source.read()
                if not self.running:
                    break
                started = time.perf_counter()
                with self.lock:
                    applied_events = self._apply_rehearsal(now)
                    row = self.pipeline.step(rgb, now)
                    self.phase = 'goal' if self.pipeline.mission.search_phase == 'goal' else 'balloon'
                    row['rehearsal'] = self.rehearsal.snapshot(now, row,
                        source=self.identities['source'], frame_sequence=self.sequence + 1)
                    row['rehearsal']['applied_event_ids'] = applied_events
                row['pipeline_ms'] = (time.perf_counter() - started) * 1000
                row['frame_time'] = now
                metadata = getattr(self.source, 'last_metadata', {})
                row['sensor_timestamp_ns'] = metadata.get('SensorTimestamp')
                row['exposure_us'] = metadata.get('ExposureTime')
                row['analogue_gain'] = metadata.get('AnalogueGain')
                row['loop_completed'] = time.monotonic()
                row['worker_errors'] = self.pipeline.worker.errors
                row['worker_restarts'] = self.pipeline.worker.restarts
                row['worker_state'] = self.pipeline.worker.state
                with self.lock:
                    self.last_frame_at = time.monotonic()
                    self.flight_telemetry = row.get('flight_telemetry')
                    self.last_row, self.last_frame_time = row, now
                    self.sequence += 1
                    self.times.append(row['loop_completed'])
                    if row['search_result'] is not None:
                        self.refresh.append(now)
                    self.ready = self.pipeline.worker.state in ('idle', 'busy') and self.pending_config is None
                    busy_before = self.capture.state in BUSY
                    copied = time.perf_counter()
                    self.capture.append(rgb, row, now)
                    if self.capture.state == 'capturing' and self.capture.frames:
                        self.capture.frames[-1][1]['record_copy_ms'] = (time.perf_counter() - copied) * 1000
                    exporting = self.capture.state == 'exporting'
                    if exporting:
                        self.ready = False
                    preview = (self.capture_mode == "legacy" and not busy_before and not exporting and now - self.preview_at >= .5
                               and now - self.watched_at < 3)
                if exporting:
                    # No image encoding/export I/O inside the timed interval.
                    if self.capture_mode == "raw":
                        if self.capture.state == "exporting" and (self.spool_thread is None or not self.spool_thread.is_alive()):
                            self.spool_thread = threading.Thread(target=self._spool, daemon=True)
                            self.spool_thread.start()
                    else:
                        self._export()
                        self.times.clear()
                        self.refresh.clear()
                elif preview:
                    with self.lock:
                        # Capture requests wait for any idle-preview work to finish.
                        if self.capture.state not in BUSY:
                            mask, _, _ = diagnostic_images(rgb, row)
                            self.frame = dict(jpeg=jpeg(rgb), mask_jpeg=jpeg(mask), sequence=self.sequence,
                                              tracks=row['tracks'], message=row['message'],
                                              flight_telemetry=row.get('flight_telemetry'))
                            self.preview_at = now
        except Exception as error:
            with self.lock:
                self.error, self.ready = f'{type(error).__name__}: {error}', False
                self.capture.finish(self.error)
        finally:
            with self.lock:
                self.capture.finish('Camera loop stopped before the take finished')
                try:
                    self.rehearsal.cancel_pending(time.monotonic())
                except (OSError, RuntimeError):
                    self.error = 'Could not persist pending rehearsal shutdown status'
            self.source.close()
            self.pipeline.worker.close()

    def _spool(self):
        try:
            result = spool_capture(self.capture, self.root, self.identities, publish=False)
            with self.lock:
                self.capture.result = result
                self.capture.frames = []
                self.capture.state = "complete"
        except Exception as error:
            with self.lock:
                self.capture.error = f"Raw spool failed: {error}"
                self.capture.state = "failed"

    def _export(self):
        try:
            summary = export_capture(self.capture, self.root, self.identities)
            with self.lock:
                self.sessions.insert(0, summary)
                self.sessions = self.sessions[:100]
        except Exception:
            # Catch even an unexpected exporter setup failure so the UI can retry.
            with self.lock:
                if self.capture.state == 'exporting':
                    self.capture.error = 'Export failed before completion'
                    self.capture.state = 'failed'


def review_page(ident):
    # All user text is assigned through textContent. Playback uses actual frame offsets.
    return '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Recorded flight take</title><style>body{font:16px system-ui;background:#edf3fa;color:#172c45;max-width:1050px;margin:32px auto;padding:16px}img{width:100%;image-rendering:auto;background:#142337}button,select,a{font:inherit;margin:8px;padding:8px}pre{white-space:pre-wrap;overflow-wrap:anywhere}input{width:100%}</style>
<a href="/">Back to Flight lab</a><h1>Recorded take</h1><p id="meta"></p>
<select id="mode" aria-label="Image view"><option value="overlay">Selection boxes</option><option value="raw">Raw camera</option><option value="mask">Search colour masks</option><option value="tracker_mask">Tracker colour bands</option></select>
<button id="play">Play</button><span id="position"></span><img id="image" alt="Recorded frame"><input id="seek" type="range" min="0" value="0" aria-label="Recorded frame">
<p>Masks are reconstructed after capture. Boxes and messages are the results recorded during the live run.</p>
<a href="download.zip">Download complete take</a><a href="summary.json">Results JSON</a><a href="wire-v1.jsonl">Exact v1 wire log</a><a href="esp32-draft.jsonl">XYZ draft log</a><a href="clip.avi" id="video">AVI preview</a><h2>Recorded frame telemetry</h2><pre id="frameTelemetry"></pre><h2>Take results</h2><pre id="result"></pre>
<script>const base='/sessions/IDENT/';let rows=[],index=0,playing=false,timer;
const $=id=>document.getElementById(id);function show(){if(!rows.length)return;$('image').src=base+rows[index][$('mode').value];$('seek').value=index;$('frameTelemetry').textContent=JSON.stringify(rows[index].flight_telemetry||{status:'Legacy take: no coordinate telemetry recorded'},null,2);$('position').textContent=`Frame ${index+1}/${rows.length}, ${rows[index].offset_s.toFixed(3)} s`;}
function tick(){show();if(!playing)return;if(index+1>=rows.length){playing=false;$('play').textContent='Play';return;}const delay=(rows[index+1].offset_s-rows[index].offset_s)*1000;index++;timer=setTimeout(tick,Math.max(1,delay));}
$('play').onclick=()=>{playing=!playing;clearTimeout(timer);$('play').textContent=playing?'Pause':'Play';if(playing){if(index===rows.length-1)index=0;tick();}};$('seek').oninput=()=>{index=+$('seek').value;show();};$('mode').onchange=show;
Promise.all([fetch(base+'summary.json').then(r=>r.json()),fetch(base+'frames.jsonl').then(r=>r.text())]).then(([s,t])=>{rows=t.trim().split('\\n').map(l=>JSON.parse(l));$('seek').max=rows.length-1;$('meta').textContent=`${s.scene} / ${s.target_label} / measured ${s.distance_m} m (lens to target centre) / ${s.frames} frames`;$('result').textContent=JSON.stringify(s,null,2);$('video').hidden=!s.video_available;show();}).catch(e=>{$('meta').textContent='Could not load take: '+e.message;});</script></html>'''.replace('IDENT', ident)


def make_handler(lab, port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, code, body, content_type='application/json'):
            if isinstance(body, (dict, list)):
                body = json.dumps(body, allow_nan=False).encode()
            elif isinstance(body, str):
                body = body.encode()
            self.send_response(code)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(body)

        def valid_host(self):
            return self.headers.get('Host') in (f'localhost:{port}', f'127.0.0.1:{port}')

        def do_GET(self):
            if not self.valid_host():
                return self.reply(403, dict(error='Use the localhost SSH tunnel'))
            path = unquote(urlsplit(self.path).path)
            if path == '/':
                return self.reply(200, PAGE.read_bytes(), 'text/html; charset=utf-8')
            if path == '/api/telemetry':
                return self.reply(200, lab.telemetry())
            if path == '/api/raw-captures':
                return self.reply(200, dict(captures=lab.raw_captures()))
            if path == '/api/state':
                return self.reply(200, lab.state())
            if path == '/api/mapping':
                telemetry = getattr(lab.pipeline, 'telemetry', None)
                if telemetry is None:
                    return self.reply(503, dict(error='Coordinate mapping is not configured'))
                return self.reply(200, telemetry.mapping())
            parts = path.strip('/').split('/')
            if len(parts) == 3 and parts[0] == 'raw-captures' and SESSION_ID.fullmatch(parts[1]) and parts[2] == 'capture.tar':
                with lab.lock:
                    if lab.capture.state in BUSY:
                        return self.reply(409, dict(error='Wait until capture/spooling finishes'))
                    lab.active_transfers += 1
                try:
                    known = {entry['id'] for entry in lab.raw_captures()}
                    file = lab.root / parts[1] / 'capture.tar'
                    if parts[1] not in known or file.parent.is_symlink():
                        return self.reply(404, dict(error='Not found'))
                    with file.open('rb') as stream:
                        self.send_response(200)
                        self.send_header('Content-Type', 'application/x-tar')
                        self.send_header('Content-Length', str(file.stat().st_size))
                        self.end_headers()
                        shutil.copyfileobj(stream, self.wfile, 64 * 1024)
                    return
                finally:
                    with lab.lock:
                        lab.active_transfers -= 1
            if len(parts) < 3 or parts[0] != 'sessions' or not SESSION_ID.fullmatch(parts[1]):
                return self.reply(404, dict(error='Not found'))
            with lab.lock:
                if lab.capture.state in BUSY:
                    return self.reply(409, dict(error='Playback and downloads pause during capture'))
                lab.active_transfers += 1
            try:
                return self.session_file(parts)
            finally:
                with lab.lock:
                    lab.active_transfers -= 1

        def session_file(self, parts):
            folder = lab.root / parts[1]
            if not (folder / 'summary.json').is_file():
                return self.reply(404, dict(error='Take is not complete'))
            if parts[2:] == ['view']:
                return self.reply(200, review_page(parts[1]), 'text/html; charset=utf-8')
            relative = Path(*parts[2:])
            allowed = (str(relative) in ('summary.json', 'scene.json', 'frames.jsonl', 'clip.avi', 'download.zip',
                                        'wire-v1.jsonl', 'wire-v2.jsonl', 'esp32-draft.jsonl', 'mapping.json',
                                        'rehearsal.jsonl', 'events.jsonl')
                       or (len(relative.parts) == 2 and relative.parts[0] in
                           ('raw', 'masks', 'overlay', 'tracker-masks')
                           and re.fullmatch(r'\d{4}\.png', relative.name)))
            file = folder / relative
            if not allowed or not file.is_file():
                return self.reply(404, dict(error='Not found'))
            types = {'.png': 'image/png', '.json': 'application/json', '.jsonl': 'application/x-ndjson',
                     '.avi': 'video/x-msvideo', '.zip': 'application/zip'}
            self.send_response(200)
            self.send_header('Content-Type', types.get(file.suffix, 'application/octet-stream'))
            self.send_header('Content-Length', str(file.stat().st_size))
            if file.suffix in ('.zip', '.avi'):
                self.send_header('Content-Disposition', f'attachment; filename="{parts[1]}{file.suffix}"')
            self.end_headers()
            with file.open('rb') as stream:
                shutil.copyfileobj(stream, self.wfile, 64 * 1024)

        def do_POST(self):
            if not self.valid_host() or self.headers.get('X-Lab-Request') != '1':
                return self.reply(403, dict(error='Invalid lab request'))
            if self.headers.get('Origin') not in (None, 'http://' + self.headers.get('Host', '')):
                return self.reply(403, dict(error='Origin does not match lab page'))
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 4096:
                    raise ValueError('Request must be at most 4096 bytes')
                payload = json.loads(self.rfile.read(size))
                if not isinstance(payload, dict):
                    raise ValueError('Settings must be an object')
                if self.path == '/api/config':
                    lab.configure(payload)
                    return self.reply(200, dict(ok=True))
                if self.path == '/api/capture':
                    return self.reply(202, lab.start_capture(payload))
                if self.path == '/api/rehearsal':
                    return self.reply(202, lab.record_rehearsal(payload))
                return self.reply(404, dict(error='Not found'))
            except (ValueError, TypeError, UnicodeDecodeError) as error:
                return self.reply(400, dict(error=str(error)))
            except RuntimeError as error:
                return self.reply(409, dict(error=str(error)))
            except OSError:
                return self.reply(503, dict(error='Could not persist rehearsal event; acceptance is unknown'))

    return Handler


class LabHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def service_actions(self):
        self.lab.check_health()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-mode', choices=('raw', 'legacy'), default='raw')
    parser.add_argument('--balloon-model', required=True)
    parser.add_argument('--goal-model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--port', type=int, default=8770)
    parser.add_argument('--balloon-limit', type=int, default=DEFAULT_BALLOON_LIMIT)
    parser.add_argument('--wall-clock-offset-s', type=float, default=0.,
                        help='Mac-derived offset for labels only; frame timing stays monotonic')
    parser.add_argument('--replay', help='Synthetic/stored frame glob for interface testing only')
    add_mapping_arguments(parser)
    args = parser.parse_args()
    if not 1 <= args.balloon_limit <= 64:
        parser.error('Balloon limit must be between 1 and 64')
    if not 1024 <= args.port <= 65535:
        parser.error('Choose a port from 1024 to 65535')
    root = Path(__file__).resolve().parents[1]
    files = [Path(args.balloon_model), Path(args.goal_model),
             Path(args.balloon_model).with_suffix('.json'), Path(args.goal_model).with_suffix('.json'),
             root / 'scripts/pi_flight_loop.py', Path(__file__), PAGE]
    files += list((root / 'src/dtr').glob('*.py'))
    files += mapping_identity_paths(args)
    telemetry = mapping_from_args(args)
    native = os.environ.get('DTR_TFLITE_LIBRARY')
    if native:
        files.append(Path(native))
    identities = dict(source='replay' if args.replay else 'camera', bundle=str(root), fps=20,
                      wall_clock_offset_s=args.wall_clock_offset_s,
                      sensor=[1296, 972], output=[320, 240], rotation_deg=180,
                      search_niceness=19, balloon_limit=args.balloon_limit,
                      mapping=telemetry.mapping(),
                      hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    worker = SearchWorker(ModelSearch(args.balloon_model, args.goal_model, args.balloon_limit), niceness=19)
    lab, thread, server, source = None, None, None, None
    try:
        worker.start()
        worker.wait_ready()
        source = Replay(args.replay, 20) if args.replay else Camera(20)
        sizes = load_target_sizes(args.target_sizes)
        # Connor's vector uses the same lens (and any checkerboard calibration) as telemetry.
        mission = Mission('orange', target_sizes=sizes,
                          vector=VectorCreator(telemetry.lens, sizes))
        lab = Lab(source, FlightPipeline(worker, ColorTracker(), mission, telemetry=telemetry),
                  args.output, identities, capture_mode=args.capture_mode)
        server = LabHTTPServer(('127.0.0.1', args.port), make_handler(lab, args.port))
        server.lab = lab
        thread = threading.Thread(target=lab.loop, daemon=True)
        thread.start()
        def stop(*_):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, stop)
        print(json.dumps(dict(url=f'http://127.0.0.1:{args.port}', source=identities['source'],
                              output=str(lab.root))), flush=True)
        server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:
        pass
    finally:
        if lab:
            lab.running = False
        if server:
            server.server_close()
        if thread:
            thread.join(10)
        if thread is None or not thread.is_alive():
            if source and lab is None:
                source.close()
            worker.close()


if __name__ == '__main__':
    main()
