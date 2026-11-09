"""Bounded flight captures. Encoding and diagnostic reconstruction happen afterward."""

from collections import Counter
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import re
import time
import uuid
import zipfile

import cv2
import numpy as np

from .flight import TRACK_HSV
from .flight import encode, encode_v2
from .goal_square import GOAL_LABELS, LABELS_BY_COLOR
from .vision import GREEN_PURPLE_PROFILE, color_masks

LABELS = {'balloon': ('green_balloon', 'purple_balloon'),
          'goal': GOAL_LABELS}
COLORS = {'green_balloon': (65, 195, 105), 'purple_balloon': (178, 100, 215),
          **{label: (245, 145, 55) for label in LABELS_BY_COLOR['orange']},
          **{label: (235, 215, 65) for label in LABELS_BY_COLOR['yellow']}}
GOAL_REQUESTS = ('orange_goal', 'yellow_goal')


def requested_target_matches(label, requested):
    """Goal capture requests cover any shape of that colour, including older takes."""
    if requested in GOAL_REQUESTS or requested in GOAL_LABELS:
        return label in LABELS_BY_COLOR[requested.split('_')[0]]
    return label == requested


def validate_capture(payload, phase, goal_color):
    if not isinstance(payload, dict):
        raise ValueError('Capture settings must be an object')
    distance = payload.get('distance_m')
    diagnostic = payload.get('diagnostic', False)
    if not isinstance(diagnostic, bool):
        raise ValueError('Diagnostic flag must be boolean')
    if not diagnostic and (isinstance(distance, bool) or not isinstance(distance, (int, float))):
        raise ValueError('Enter the tape-measured distance in metres')
    if not diagnostic and (not math.isfinite(distance) or not 0 < distance <= 100):
        raise ValueError('Measured distance must be greater than zero and at most 100 m')
    label = payload.get('target_label')
    if label not in (LABELS.get(phase, ()) + (GOAL_REQUESTS if phase == 'goal' else ())):
        raise ValueError('Target label does not match the search phase')
    if phase == 'goal' and label not in (*LABELS_BY_COLOR[goal_color], goal_color + '_goal'):
        raise ValueError('Goal label must match the selected goal colour')
    count = payload.get('expected_count', 1)
    if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= 20:
        raise ValueError('Expected target count must be an integer from 0 to 20')
    scene, notes = payload.get('scene', ''), payload.get('notes', '')
    if not isinstance(scene, str) or (scene and not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,47}', scene)):
        raise ValueError('Scene name: up to 48 lowercase letters, digits, - or _')
    if not isinstance(notes, str) or len(notes) > 500:
        raise ValueError('Notes are limited to 500 characters')
    purpose = payload.get('data_purpose', 'development')
    if purpose not in ('development', 'training'):
        raise ValueError('Data purpose must be development or training')
    if diagnostic and purpose != 'development':
        raise ValueError('Unmeasured diagnostic captures are development only')
    group = payload.get('session_group', '')
    if not isinstance(group, str) or (group and not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,63}', group)):
        raise ValueError('Session group: up to 64 lowercase letters, digits, - or _')
    if purpose == 'training' and not group:
        raise ValueError('Training collection requires a session group to prevent frame split leakage')
    return dict(distance_m=None if diagnostic else float(distance),
                distance_source='not measured' if diagnostic else 'operator tape measurement',
                distance_reference='camera lens to target centre', target_label=label,
                expected_count=None if diagnostic else count,
                diagnostic=diagnostic,
                scene=scene or (f'{phase}-diagnostic' if diagnostic else f'{phase}-{distance:g}m'), notes=notes,
                phase=phase, goal_color=goal_color, training_use=purpose == 'training',
                data_purpose=purpose, session_group=group or None,
                evaluation_use='training collection; review required before use' if purpose == 'training'
                else 'development lab; not flight qualification')


class CaptureBuffer:
    def __init__(self, duration_s=5.0, max_frames=140, max_bytes=40 * 1024 * 1024,
                 wall_clock_offset_s=0.):
        self.duration_s, self.max_frames, self.max_bytes = duration_s, max_frames, max_bytes
        self.wall_clock_offset_s = wall_clock_offset_s
        self.state, self.id, self.error = 'idle', None, None
        self.frames, self.metadata, self.result = [], {}, None
        self.events = []
        self.started, self.bytes_used, self.count = 0., 0, 0
        self.finished_at = None

    def start(self, metadata, now):
        if self.state in ('capturing', 'exporting'):
            raise ValueError('A capture is already running')
        wall_now = datetime.now(timezone.utc) + timedelta(seconds=self.wall_clock_offset_s)
        self.id = wall_now.strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex[:8]
        self.metadata = deepcopy(metadata)
        self.metadata['created_at'] = wall_now.isoformat()
        self.metadata['wall_clock_offset_s'] = self.wall_clock_offset_s
        self.frames, self.error, self.result = [], None, None
        self.events = []
        self.started, self.bytes_used, self.count = now, 0, 0
        self.finished_at = None
        self.state = 'capturing'
        return self.id

    def record_event(self, event):
        stamp = event.get('applied_at_monotonic', event['received_at_monotonic'])
        if self.state != 'capturing' or not self.started <= stamp < self.started + self.duration_s:
            return
        if len(self.events) >= 512:
            raise RuntimeError('Take event limit reached; wait until this take is saved')
        self.events.append(deepcopy(event))

    def append(self, rgb, row, now):
        if self.state != 'capturing':
            return False
        if now >= self.started + self.duration_s:
            self.finish(now=now)
            return False
        if now < self.started:
            return False  # A frame already in flight when the operator pressed Capture.
        if self.frames and now <= self.frames[-1][1]['frame_time']:
            self.finish('Non-increasing capture timestamps')
            return False
        if len(self.frames) >= self.max_frames or self.bytes_used + rgb.nbytes > self.max_bytes:
            self.finish('Capture memory limit exceeded; this take is incomplete')
            return False
        saved = deepcopy(row)
        saved['frame_time'] = float(now)
        saved['offset_s'] = float(now - self.started)
        self.frames.append((rgb.copy(), saved))
        self.bytes_used += rgb.nbytes
        self.count = len(self.frames)
        return True

    def finish(self, error=None, now=None):
        if self.state != 'capturing':
            return
        self.error = error or (None if self.frames else 'No camera frames were captured')
        self.finished_at = time.monotonic() if now is None else now
        self.state = 'failed' if self.error else 'exporting'

    def status(self, now=None):
        now = time.monotonic() if now is None else now
        return dict(state=self.state, id=self.id, frames=self.count, error=self.error,
                    remaining_s=max(0., self.started + self.duration_s - now)
                    if self.state == 'capturing' else 0., result=self.result)


def diagnostic_images(rgb, row):
    """Reconstruct the unchanged pipeline's colour masks; never re-run inference."""
    phase = 'goal' if row['phase_before'] == 'goal' else 'balloon'
    masks = color_masks(rgb, phase, GREEN_PURPLE_PROFILE if phase == 'balloon' else 'v2')
    mask_rgb = np.zeros_like(rgb)
    mask_labels = ('orange_circle', 'yellow_circle') if phase == 'goal' else LABELS[phase]
    for label, mask in zip(mask_labels, masks):
        mask_rgb[mask != 0] = COLORS[label]
    overlay = rgb.copy()
    message = row['message']
    for track in row['tracks']:
        x1, y1, x2, y2 = map(lambda v: int(round(v)), track['box'])
        colour = COLORS.get(track['label'], (180, 180, 180))
        cv2.rectangle(overlay, (x1, y1), (x2, y2), colour, 1)
        cv2.putText(overlay, track['label'].replace('_', ' '), (x1, max(10, y1 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, .28, colour, 1, cv2.LINE_AA)
    target = message.get('target')
    if target:
        cx, cy = int((target['x'] + 1) * rgb.shape[1] / 2), int((target['y'] + 1) * rgb.shape[0] / 2)
        cv2.drawMarker(overlay, (cx, cy), (80, 210, 255), cv2.MARKER_CROSS, 12, 2)
    cv2.putText(overlay, message['cmd'], (6, rgb.shape[0] - 7),
                cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1, cv2.LINE_AA)
    delta = row.get('flight_telemetry', {}).get('outgoing', {}).get('draft_packet', {}).get('navigation_delta_m')
    if delta:
        text = 'Est XYZ ' + ', '.join(f'{delta[key]:.2f}' for key in
                                     ('x_right_m', 'y_up_m', 'z_forward_m')) + ' m'
        cv2.putText(overlay, text, (6, rgb.shape[0] - 24),
                    cv2.FONT_HERSHEY_SIMPLEX, .33, (255, 255, 255), 1, cv2.LINE_AA)
    # Tracker uses local colour windows. These full-frame images use those same bands,
    # and are labelled diagnostic reconstructions rather than stored internal masks.
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    tracker_mask = np.zeros(rgb.shape[:2], np.uint8)
    for track in row['tracks']:
        for lo, hi in TRACK_HSV[track['label']]:
            tracker_mask |= cv2.inRange(hsv, lo, hi)
    return mask_rgb, overlay, tracker_mask


def _write_png(path, rgb):
    pixels = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) if rgb.ndim == 3 else rgb
    if not cv2.imwrite(str(path), pixels, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
        raise OSError(f'Failed to save {path.name}')


def export_capture(capture, root, identities):
    if capture.state != 'exporting' or not capture.frames:
        raise ValueError('Only a finished, nonempty capture can be exported')
    root = Path(root)
    pending, final = root / ('.' + capture.id + '.partial'), root / capture.id
    writer = None
    try:
        root.mkdir(parents=True, exist_ok=True)
        pending.mkdir()
        for name in ('raw', 'masks', 'overlay', 'tracker-masks'):
            (pending / name).mkdir()
        rows = [row for _, row in capture.frames]
        times = [row['frame_time'] for row in rows]
        span = times[-1] - times[0]
        fps = (len(times) - 1) / span if span > 0 else None
        if fps:
            h, w = capture.frames[0][0].shape[:2]
            writer = cv2.VideoWriter(str(pending / 'clip.avi'), cv2.VideoWriter_fourcc(*'MJPG'),
                                     fps, (w, h))
            if not writer.isOpened():
                writer.release()
                writer = None
                (pending / 'clip.avi').unlink(missing_ok=True)
        commands = Counter()
        matching = 0
        wire_bytes, draft_bytes, v2_bytes = [], [], []
        with (pending / 'events.jsonl').open('x') as events:
            for event in capture.events:
                events.write(json.dumps(event, allow_nan=False) + '\n')
        with (pending / 'rehearsal.jsonl').open('x') as rehearsal:
            for row in rows:
                if 'rehearsal' in row:
                    rehearsal.write(json.dumps(dict(frame_time=row['frame_time'],
                                                     **row['rehearsal']), allow_nan=False) + '\n')
        # Serialize only observations already computed in the live frame step.
        # Old takes without coordinate telemetry remain explicitly legacy-only.
        with (pending / 'wire-v1.jsonl').open('x') as wire, (pending / 'wire-v2.jsonl').open('x') as wire2, (pending / 'esp32-draft.jsonl').open('x') as draft:
            for row in rows:
                outgoing = row.get('flight_telemetry', {}).get('outgoing', {})
                legacy_line = outgoing.get('legacy_jsonl', encode(row['message']))
                v2_line = outgoing.get('wire_v2_jsonl', encode_v2(row['message']))
                wire2.write(v2_line)
                v2_bytes.append(len(v2_line.encode('utf-8')))
                wire.write(legacy_line)
                wire_bytes.append(len(legacy_line.encode('utf-8')))
                if outgoing.get('draft_jsonl') is not None:
                    draft.write(outgoing['draft_jsonl'])
                    draft_bytes.append(len(outgoing['draft_jsonl'].encode('utf-8')))
        if 'mapping' in identities:
            (pending / 'mapping.json').write_text(json.dumps(identities['mapping'], indent=2, allow_nan=False) + '\n')
        with (pending / 'frames.jsonl').open('x') as stream:
            for index, (rgb, row) in enumerate(capture.frames):
                filename = f'{index:04d}.png'
                mask, overlay, tracker_mask = diagnostic_images(rgb, row)
                for name, pixels in (('raw', rgb), ('masks', mask), ('overlay', overlay),
                                     ('tracker-masks', tracker_mask)):
                    _write_png(pending / name / filename, pixels)
                if writer is not None:
                    writer.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                stream.write(json.dumps(dict(row, raw=f'raw/{filename}', mask=f'masks/{filename}',
                                            overlay=f'overlay/{filename}',
                                            tracker_mask=f'tracker-masks/{filename}'), allow_nan=False) + '\n')
                commands[row['message']['cmd']] += 1
                matching += any(requested_target_matches(t['label'], capture.metadata['target_label'])
                                and t['misses'] == 0
                                for t in row['tracks'])
        if writer is not None:
            writer.release()
            writer = None
        gaps = np.diff(times) * 1000
        deadline = capture.started + capture.duration_s
        start_gap = max(0., times[0] - capture.started)
        end_gap = max(0., deadline - times[-1])
        termination_delay = max(0., capture.finished_at - deadline)
        largest_gap = max(start_gap, end_gap, termination_delay,
                          float(gaps.max()) / 1000 if len(gaps) else 0.)
        window_ok = largest_gap <= .25 and capture.finished_at >= deadline
        capture_error = None if window_ok else 'Interrupted recording: a window boundary or frame gap exceeded 250 ms'
        summary = dict(capture.metadata, id=capture.id, frames=len(rows), complete=window_ok,
                       export_complete=True, capture_error=capture_error,
                       capture_started_monotonic=capture.started,
                       capture_finished_monotonic=capture.finished_at,
                       start_gap_ms=start_gap * 1000, end_gap_ms=end_gap * 1000,
                       termination_delay_ms=termination_delay * 1000,
                       max_window_gap_ms=largest_gap * 1000,
                       continuity_limit_ms=250,
                       continuity_note='Recording integrity check; not a flight acceptance threshold',
                       requested_duration_s=capture.duration_s, sample_span_s=span,
                       loop_fps=fps, max_gap_ms=float(gaps.max()) if len(gaps) else None,
                       p95_gap_ms=float(np.percentile(gaps, 95)) if len(gaps) else None,
                       commands=dict(commands), frames_with_requested_target=matching,
                       rehearsal=dict(events='events.jsonl', frames='rehearsal.jsonl',
                                      event_records=len(capture.events),
                                      transmitted=False, applied=False,
                                      possession_source='manual operator observations only'),
                       worker_errors_after=rows[-1].get('worker_errors'),
                       worker_restarts_after=rows[-1].get('worker_restarts'),
                       requested_target_coverage=matching / len(rows),
                       coverage_note='Tracker presence, not reviewed accuracy or recall',
                       masks_note='Colour masks reconstructed after capture using the same flight functions',
                       preview_during_capture=False, encoding_during_capture=False,
                       video_available=(pending / 'clip.avi').exists(),
                       video_note='AVI uses measured average fps; frames.jsonl preserves exact timestamps',
                       outgoing_logs=dict(wire_v2='wire-v2.jsonl', wire_v2_max_bytes=max(v2_bytes, default=0), legacy='wire-v1.jsonl', esp32_draft='esp32-draft.jsonl',
                                          draft_frames=sum('flight_telemetry' in row for row in rows),
                                          transmitted=False, controller_input='unavailable',
                                          max_legacy_line_bytes=max(wire_bytes, default=0),
                                          max_draft_line_bytes=max(draft_bytes, default=0),
                                          draft_min_bits_s_at_20hz_8n1=max(draft_bytes, default=0) * 20 * 10,
                                          link_rate_status='not negotiated; rich draft is not a final UART contract'),
                       identities=identities, summary_url=f'/sessions/{capture.id}/summary.json',
                       view_url=f'/sessions/{capture.id}/view',
                       download_url=f'/sessions/{capture.id}/download.zip')
        (pending / 'scene.json').write_text(json.dumps(capture.metadata, indent=2) + '\n')
        (pending / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
        with zipfile.ZipFile(pending / 'download.zip', 'x', compression=zipfile.ZIP_STORED) as archive:
            for path in sorted(pending.rglob('*')):
                if path.is_file() and path.name != 'download.zip':
                    archive.write(path, path.relative_to(pending))
        pending.rename(final)
        capture.result = summary
        capture.error = capture_error
        capture.state = 'complete' if window_ok else 'failed'
        capture.frames = []
        return summary
    except Exception as error:
        capture.error, capture.state = f'Export failed: {error}', 'failed'
        raise
    finally:
        if writer is not None:
            writer.release()
