"""Per-frame color tracker, mission phases and the Pi -> ESP32 message.

Guidance only: the ESP32 owns flight control, arena limits and failsafes.
Full-search results arrive late (about 0.5 s on the Pi); the tracker matches
them against where each track was when that frame was captured.
"""

from collections import deque
from dataclasses import dataclass, field
import json
import struct
import math

import cv2
import numpy as np

from .lens import LensModel
from .target_vector import VectorCreator
from .goal_square import GOAL_COLORS, GOAL_LABELS, LABELS_BY_COLOR
from .target_geometry import load_target_sizes, relative_range
from .vision import GREEN_PURPLE_HSV, HSV_RANGES

BALLOON_LABELS = ("green_balloon", "purple_balloon")
TRACK_HSV = {
    "green_balloon": GREEN_PURPLE_HSV[:1],
    "purple_balloon": GREEN_PURPLE_HSV[1:],
    # Same bands the goal search uses; orange also wraps through red hues.
    **{label: (HSV_RANGES["goal"][0], ((170, 45, 30), (179, 255, 255)))
       for label in LABELS_BY_COLOR["orange"]},
    **{label: (HSV_RANGES["goal"][1],) for label in LABELS_BY_COLOR["yellow"]},
}
PHASE_LABELS = {"balloon": BALLOON_LABELS, "capture": BALLOON_LABELS,
                "goal": GOAL_LABELS}
LABELS = BALLOON_LABELS + tuple(GOAL_LABELS)
PHASES = ("balloon", "capture", "goal")
COMMANDS = ("SEARCH", "TRACK", "LOST", "FAN_ON", "ENTER", "AVOID")
PROTOCOL_VERSION = 1
MIN_CENTER_FRACTION = 0.6  # Below this centre-line/box height ratio, trust the box height.


def iou(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    inter = max(0.0, w) * max(0.0, h)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def same_target(a, b):
    """Same tracked object identity: equal labels, or one goal colour whose shape was revised."""
    return a == b or (a in GOAL_LABELS and b in GOAL_LABELS
                      and a.split("_")[0] == b.split("_")[0])


def center(box):
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def side(box):
    return max(box[2] - box[0], box[3] - box[1])


def dims(box):
    return box[2] - box[0], box[3] - box[1]


@dataclass
class Track:
    id: int
    label: str
    box: list
    confirmed: float  # Capture time of the newest frame a full search confirmed it in.
    seen: float  # Time of the last color hit (or confirmation).
    score: float = 0.0
    velocity: tuple = (0.0, 0.0)  # Pixels per second.
    misses: int = 0
    history: deque = field(default_factory=lambda: deque(maxlen=40))
    confirmed_wh: tuple = (0.0, 0.0)  # Width and height the full search measured, in pixels.

    last_miss: float = float("-inf")  # Last miss or continuity break (jump to another blob).
    follow_start: float = float("inf")
    center_h: float | None = None

    def box_at(self, when):
        """Where the track was at `when`; the oldest kept box if `when` predates it."""
        best = self.box
        for stamp, box in reversed(self.history):
            best = box
            if stamp <= when:
                break
        return best


class ColorTracker:
    def __init__(self, width=320, height=240, min_pixels=4, max_misses=4, confirm_timeout=2.5,
                 window_scale=2.5, min_window=24, max_tracks=6, alpha=0.5, max_growth=1.6,
                 min_match_iou=0.3, max_size_ratio=1.5, follow_jump=0.5):
        self.width, self.height = width, height
        # A hit this far (x target size) from the predicted centre may be a different
        # same-colour object: keep tracking, but require a new full-search confirmation
        # before Mission treats the track as followed (see Mission.freshness).
        self.follow_jump = follow_jump
        self.min_pixels, self.max_misses = min_pixels, max_misses
        self.confirm_timeout = confirm_timeout
        self.window_scale, self.min_window = window_scale, min_window
        self.max_tracks, self.alpha = max_tracks, alpha
        # Per-hit size change limit: real approach grows a target slowly at 20 fps, so a
        # sudden jump means the window latched onto other same-colored clutter.
        self.max_growth = max_growth
        self.min_match_iou, self.max_size_ratio = min_match_iou, max_size_ratio
        self.tracks, self.next_id, self.last_update = [], 1, None

    def clear(self):
        self.tracks = []

    def _window(self, track, dt):
        cx, cy = center(track.box)
        cx, cy = cx + track.velocity[0] * dt, cy + track.velocity[1] * dt
        w, h = track.box[2] - track.box[0], track.box[3] - track.box[1]
        # Search wider after misses so a fast or late-confirmed target is re-found.
        half = max(self.min_window, self.window_scale * max(w, h) * (1 + 0.5 * track.misses)) / 2
        x1, y1 = int(max(0, cx - half)), int(max(0, cy - half))
        x2, y2 = int(min(self.width, cx + half)), int(min(self.height, cy + half))
        return (x1, y1, x2, y2), (cx, cy)

    def _hit(self, hsv, label, window, predicted, size, growth):
        """Box of the label-colored blob nearest the predicted center, or None."""
        x1, y1, x2, y2 = window
        if x2 - x1 < 2 or y2 - y1 < 2:
            return None
        patch = hsv[y1:y2, x1:x2]
        mask = np.zeros(patch.shape[:2], np.uint8)
        for lo, hi in TRACK_HSV[label]:
            mask |= cv2.inRange(patch, lo, hi)
        count, components, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        best, best_distance = None, None
        for index in range(1, count):
            if stats[index, cv2.CC_STAT_AREA] < self.min_pixels:
                continue
            side = max(stats[index, cv2.CC_STAT_WIDTH], stats[index, cv2.CC_STAT_HEIGHT])
            if not size / growth <= side <= size * growth + 2:
                continue
            distance = np.hypot(centroids[index][0] + x1 - predicted[0],
                                centroids[index][1] + y1 - predicted[1])
            if best is None or distance < best_distance:
                best, best_distance = index, distance
        if best is None:
            return None
        self._hit_distance = float(best_distance)
        bx, by, bw, bh = stats[best, :4]
        mid, band = int(bx + bw // 2), max(1, round(0.05 * bw))
        rows = np.flatnonzero(np.any(components[:, max(0, mid-band):mid+band+1] == best, axis=1))
        self._center_h = float(rows[-1] - rows[0] + 1) if len(rows) else float(bh)
        return [float(x1 + bx), float(y1 + by), float(x1 + bx + bw), float(y1 + by + bh)]

    def update(self, rgb, now):
        """Follow every track into this frame with a cheap color check."""
        if rgb.shape[:2] != (self.height, self.width):
            raise ValueError("Frame size does not match the tracker")
        dt = 0.0 if self.last_update is None else max(0.0, now - self.last_update)
        self.last_update = now
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV) if self.tracks else None
        kept = []
        for track in self.tracks:
            window, predicted = self._window(track, dt)
            size = max(track.box[2] - track.box[0], track.box[3] - track.box[1])
            # Allow more scale change after misses (e.g. a snap to a late, smaller box);
            # Mission still refuses to act on growth the full search has not confirmed.
            growth = self.max_growth * (1 + 0.25 * track.misses)
            box = self._hit(hsv, track.label, window, predicted, size, growth)
            if box is None:
                track.misses += 1
                track.last_miss = now
                dx, dy = track.velocity[0] * dt, track.velocity[1] * dt
                track.box = [track.box[0] + dx, track.box[1] + dy, track.box[2] + dx, track.box[3] + dy]
            else:
                track.center_h = self._center_h
                if self._hit_distance > self.follow_jump * max(size, 1.0):
                    track.last_miss = now  # Continuity break, not a miss: tracking goes on.
                if dt > 0:
                    (ox, oy), (nx, ny) = center(track.box), center(box)
                    a = self.alpha
                    track.velocity = (a * (nx - ox) / dt + (1 - a) * track.velocity[0],
                                      a * (ny - oy) / dt + (1 - a) * track.velocity[1])
                track.box, track.seen, track.misses = box, now, 0
            track.history.append((now, list(track.box)))
            if track.misses <= self.max_misses and now - track.confirmed <= self.confirm_timeout:
                kept.append(track)
        self.tracks = kept
        return self.tracks

    def merge(self, detections, frame_time, now):
        """Fold in full-search detections made on the frame captured at `frame_time`."""
        unmatched = []
        claimed = set()
        for det in sorted(detections, key=lambda d: -d.get("score", 0.0)):
            det_size = side(det["box"])
            dx, dy = center(det["box"])
            best, best_iou, best_distance = None, -1.0, float("inf")
            for track in self.tracks:
                if not same_target(track.label, det["label"]) or track.id in claimed:
                    continue
                then = track.box_at(frame_time)
                overlap = iou(then, det["box"])
                tx, ty = center(then)
                distance = np.hypot(tx - dx, ty - dy)
                if overlap < 0.1 and distance > det_size:
                    continue
                if (overlap, -distance) > (best_iou, -best_distance):
                    best, best_iou, best_distance = track, overlap, distance
            if best is None:
                unmatched.append(det)
                continue
            claimed.add(best.id)
            if frame_time < best.confirmed:
                continue  # A newer search result already confirmed or corrected this track.
            # A newer full search may revise a goal's shape; one hoop keeps one track.
            best.label = det["label"]
            # Compare width and height separately: a box can swell along one axis only.
            ratio = max(max(a, b) / max(min(a, b), 1)
                        for a, b in zip(dims(best.box_at(frame_time)), dims(det["box"])))
            consistent = (ratio <= self.max_size_ratio
                          and (best_iou >= self.min_match_iou
                               or (det_size < 12 and best_distance <= 0.5 * det_size)))
            if consistent:
                # Keep the fresher color box only when historical geometry agrees.
                best.confirmed = frame_time
            else:
                # The classifier wins; motion from the drifting color track is unsafe.
                best.box = [float(v) for v in det["box"]]
                best.velocity, best.misses = (0.0, 0.0), 1
                best.follow_start = now
                best.confirmed = frame_time
                best.history.append((now, list(best.box)))
            best.confirmed_wh = dims(det["box"])
            best.score = float(det.get("score", best.score))
        for det in unmatched:
            if len(self.tracks) >= self.max_tracks:
                break
            box = [float(v) for v in det["box"]]
            # Start where the object was; widened windows re-find it in the next frames.
            track = Track(self.next_id, det["label"], box, confirmed=frame_time, seen=frame_time,
                          score=float(det.get("score", 0.0)), misses=1, follow_start=now, confirmed_wh=dims(box), center_h=box[3]-box[1])
            track.history.append((frame_time, list(box)))
            self.tracks.append(track)
            self.next_id += 1
        self._drop_duplicates()
        return self.tracks

    def _drop_duplicates(self):
        kept = []
        # Newest confirmation wins so a classifier correction is never discarded.
        for track in sorted(self.tracks, key=lambda t: (-t.confirmed, t.misses)):
            if all(not same_target(t.label, track.label) or iou(t.box, track.box) < 0.5
                   for t in kept):
                kept.append(track)
        self.tracks = sorted(kept, key=lambda t: t.id)


@dataclass
class MissionConfig:
    fresh_s: float = 0.8  # FAN_ON / ENTER need a full-search confirmation this recent.
    follow_s: float = 2.5
    lost_s: float = 2.0  # Send LOST (hold) this long after losing the target, then SEARCH.
    enter_commit_s: float = 3.0  # Keep ENTER this long after the gate last passed (hoop clips/leaves view).
    fan_size: float = 0.30  # Balloon width / frame width to trigger FAN_ON (provisional).
    enter_size: float = 0.45  # Square width / frame width; other goal widths scale this gate.
    centered: float = 0.25  # |x| and |y| limit for FAN_ON / ENTER.
    max_growth: float = 1.5  # Current side / full-search side limit for FAN_ON / ENTER.
    capture_timeout_s: float = 4.0  # No `captured` event in time: back to balloon.


class Mission:
    """balloon -> capture (FAN_ON) -> goal -> scored -> balloon. Emits one message per frame."""

    def __init__(self, goal_color, width=320, height=240, config=None, sizes_path=None,
                 target_sizes=None, vector=None):
        self.config = config or MissionConfig()
        self.target_sizes = (target_sizes if target_sizes is not None
                             else load_target_sizes(sizes_path))
        self.vector = vector or VectorCreator(LensModel.ideal_equidistant(width, height), self.target_sizes)
        self.last_description, self.last_cmd, self.search_dir = None, None, 0
        self.enter_at, self.enter_track_id = None, None
        self.width, self.height = width, height
        self.set_goal_color(goal_color)
        self.phase, self.fan, self.seq = "balloon", False, 0
        self.last_target_at, self.capture_started = None, None
        self.selected_track_id = None

    def set_goal_color(self, color):
        if color not in GOAL_COLORS:
            raise ValueError(f"Goal color must be one of {GOAL_COLORS}")
        self.goal_color = color

    @property
    def search_phase(self):
        return self.phase

    def _enter(self, phase):
        self.phase, self.last_target_at = phase, None
        self.last_description, self.last_cmd, self.search_dir = None, None, 0
        self.enter_at, self.enter_track_id = None, None

    def captured(self):
        if self.phase in ("balloon", "capture"):
            self.fan = True
            self._enter("goal")

    def capture_failed(self):
        self.fan = False
        self._enter("balloon")

    def scored(self):
        if self.phase == "goal":
            self.fan = False
            self._enter("balloon")

    def describe(self, track, now):
        x1, y1, x2, y2 = track.box
        cx, cy = center(track.box)
        # Centre-line height corrects fisheye/yaw bending by a modest fraction; far less
        # than the box means the column missed part of a hollow frame, so use the box.
        height = track.center_h
        if height is None or height < MIN_CENTER_FRACTION * (y2 - y1):
            height = None
        size = self.target_sizes.get(track.label, {})
        if track.label in BALLOON_LABELS and size.get("height_m"):
            # A specular highlight on top drops out of the colour mask, so the
            # balloon's vertical extent reads short. Its width, scaled to height
            # by the measured shape, is the steadier cue: take the larger.
            height = max(height or 0.0, y2 - y1, (x2 - x1) * size["height_m"] / size["width_m"])
        vector = self.vector.vector_for(track.label, track.box, height) or {}
        return dict(**{k: round(vector[k], digits) if k in vector else None
                       for k, digits in (("yaw_deg", 1), ("elev_m", 2), ("travel_m", 2))},
                    label=track.label,
                    x=round(2 * cx / self.width - 1, 3), y=round(2 * cy / self.height - 1, 3),
                    size=round((x2 - x1) / self.width, 3),
                    age_ms=int(max(0.0, now - track.confirmed) * 1000),
                    seen_ms=int(max(0.0, now - track.seen) * 1000),
                    follow=(now - track.confirmed > self.config.fresh_s
                            and self.freshness({"age_ms": int(max(0, now-track.confirmed)*1000)}, track)))

    def freshness(self, target, track):
        age = target["age_ms"]
        recent = age <= self.config.fresh_s * 1000
        followed = (age <= self.config.follow_s * 1000 and track.misses == 0
                    and track.last_miss <= track.confirmed
                    and track.follow_start <= track.confirmed)
        return recent or followed

    def _ready(self, target, track, size_limit=None, range_limit=None):
        c = self.config
        # A color box swollen past what the full search measured may be clutter.
        grown = all(now <= c.max_growth * then for now, then in zip(dims(track.box), track.confirmed_wh))
        if range_limit is not None:
            # Goals: the same yaw-tolerant range proxy that picks the nearest hoop.
            near = relative_range(track.label, track.box, self.target_sizes) <= range_limit
        else:
            # Use the measured width, not the rounded telemetry value, at the size gate.
            near = (track.box[2] - track.box[0]) / self.width >= size_limit
        return (grown and self.freshness(target, track) and near
                and abs(target["x"]) <= c.centered and abs(target["y"]) <= c.centered)

    def _best(self, tracks, labels, now):
        found = [t for t in tracks if t.label in labels and t.misses == 0]
        if not found:
            return None
        if self.phase == "goal":
            return min(found, key=lambda t: (relative_range(t.label, t.box, self.target_sizes),
                                             -t.confirmed))
        # Balloons share a physical size: biggest first, then newest confirmation.
        return max(found, key=lambda t: (t.box[2] - t.box[0], t.confirmed))

    def _committed(self, now):
        return self.enter_at is not None and now - self.enter_at < self.config.enter_commit_s

    def step(self, tracks, now):
        c = self.config
        cmd, target = None, None
        self.selected_track_id = None
        if self.phase in ("balloon", "capture"):
            track = self._best(tracks, BALLOON_LABELS, now)
            target = self.describe(track, now) if track else None
            self.selected_track_id = track.id if track else None
            if self.phase == "capture":
                # `fan` stays on as state; the command still needs a ready target.
                if now - self.capture_started > c.capture_timeout_s:
                    self.capture_failed()
                elif target and self._ready(target, track, c.fan_size):
                    cmd = "FAN_ON"
            elif target and self._ready(target, track, c.fan_size):
                self.fan, self.capture_started, cmd = True, now, "FAN_ON"
                self._enter("capture")
        else:
            committed = self._committed(now)
            # A pass-through stays bound to the hoop that passed the gate.
            held = next((t for t in tracks if t.id == self.enter_track_id and t.misses == 0),
                        None) if committed else None
            ours = held or self._best(tracks, LABELS_BY_COLOR[self.goal_color], now)
            if ours:
                candidate = self.describe(ours, now)
                # Every shape enters at the range where a square spans enter_size of the frame.
                range_limit = (self.target_sizes[f"{self.goal_color}_square"]["width_m"]
                               / (c.enter_size * self.width))
                # Up close the committed hoop clips and swells past the gate; stay committed.
                ready = self._ready(candidate, ours, range_limit=range_limit)
                if ready:
                    self.enter_at, self.enter_track_id, committed = now, ours.id, True
                if ready or ours is held or not committed:
                    self.selected_track_id = ours.id
                    target = candidate
                    cmd = "ENTER" if ready or ours is held else "TRACK"
                # Otherwise the committed hoop is out of view: keep its frozen ENTER below.
            if target is None and not committed:
                selected = self._best(tracks, tuple(label for label in GOAL_LABELS
                                      if label not in LABELS_BY_COLOR[self.goal_color]), now)
                if selected:
                    self.selected_track_id = selected.id
                    target, cmd = self.describe(selected, now), "AVOID"
        if cmd is None:
            if target:
                cmd = "TRACK"
            elif self.last_target_at is not None and now - self.last_target_at < c.lost_s:
                cmd = "LOST"
            else:
                cmd = "SEARCH"
        lost_ms = None
        if target and cmd != "AVOID":
            self.last_target_at = now
            self.last_description = dict(target)
            self.last_cmd = cmd
            self.search_dir = 0
        elif target is None and self.last_target_at is not None:
            elapsed = now - self.last_target_at
            lost_ms = int(max(0, elapsed) * 1000)
            yaw = (self.last_description or {}).get('yaw_deg')
            self.search_dir = (1 if yaw > 0 else -1) if yaw is not None and abs(yaw) >= 2 else 0
            if self._committed(now):
                cmd = 'ENTER'
            if cmd in ('LOST', 'ENTER'):
                # Frozen sighting: it ages with the loss and is never "followed".
                target = dict(self.last_description, follow=False)
                for key in ('age_ms', 'seen_ms'):
                    if target.get(key) is not None:
                        target[key] += lost_ms
        self.seq += 1
        return dict(v=PROTOCOL_VERSION, seq=self.seq, t_ms=int(now * 1000), phase=self.phase,
                    goal_color=self.goal_color, cmd=cmd, fan=self.fan, target=target,
                    lost_ms=lost_ms, search_dir=self.search_dir)


def encode(message):
    """One compact JSON line, the wire format for the ESP32 link."""
    if message["cmd"] not in COMMANDS:
        raise ValueError("Unknown command")
    return json.dumps(message, separators=(",", ":")) + "\n"


def encode_v2(message):
    """Compact ESP32 guidance. t_ms - lost_ms is the frozen sighting time;
    the receiver can subtract its IMU yaw change since that sighting.
    """
    if message['cmd'] not in COMMANDS:
        raise ValueError('Unknown command')
    target = message.get('target') or {}
    if message['cmd'] == 'SEARCH':
        target = {}
    packet = dict(v=2, **{k: message[k] for k in
                          ('seq', 't_ms', 'cmd', 'phase', 'goal_color', 'fan')})
    packet.update({k: target.get(k) for k in ('label', 'yaw_deg', 'elev_m', 'travel_m', 'age_ms')})
    packet.update(lost_ms=message.get('lost_ms'), search_dir=message.get('search_dir', 0))
    return json.dumps(packet, separators=(',', ':'), allow_nan=False) + '\n'


def crc8(data):
    """CRC-8, polynomial 0x07, initial value zero."""
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ (7 if crc & 128 else 0)) & 255
    return crc


def encode_i2c(message):
    """Fixed 20-byte little-endian guidance packet (no I2C register prefix).
    Offset/type: 0 u8 magic D7; 1 u8 version 2; 2 u16 sequence;
    4 u8 COMMANDS index; 5 u8 PHASES index; 6 u8 LABELS index (FF none);
    7 u8 flags: bit0 fan, bit1 yellow goal, bit2 target, bit3 follow freshness;
    8 i16 yaw centidegrees (+right); 10 i16 elevation cm (+climb);
    12 u16 travel cm (FFFF none); 14 u16 age ms; 16 u16 lost ms;
    18 i8 search direction; 19 u8 CRC-8 over bytes 0..18 (poly 07, init 00).
    Age/lost saturate at FFFE; absent times also use FFFE. Other numbers
    saturate to their field range. Enum order is defined by the module tuples.
    """
    def number(value, low, high, scale=1):
        try:
            value = float(value) * scale
            if math.isnan(value):
                return low if low >= 0 else 0
            return int(max(low, min(high, round(value) if math.isfinite(value) else value)))
        except (TypeError, ValueError, OverflowError):
            return low if low >= 0 else 0
    def index(values, value, default=0):
        return values.index(value) if value in values else default
    target = (message.get("target") or {}) if message.get("cmd") != "SEARCH" else {}
    flags = (bool(message.get("fan")) | (int(message.get("goal_color") == "yellow") << 1)
             | (int(bool(target)) << 2) | (int(bool(target.get("follow"))) << 3))
    travel = target.get("travel_m")
    body = struct.pack("<BBHBBBBhhHHHb", 0xD7, 2,
        number(message.get("seq"), 0, 65535), index(COMMANDS, message.get("cmd")),
        index(PHASES, message.get("phase")), index(LABELS, target.get("label"), 255), flags,
        number(target.get("yaw_deg"), -32768, 32767, 100),
        number(target.get("elev_m"), -32768, 32767, 100),
        65535 if travel is None else number(travel, 0, 65534, 100),
        number(target.get("age_ms", 65534), 0, 65534),
        number(message.get("lost_ms") if message.get("lost_ms") is not None else 65534, 0, 65534),
        number(message.get("search_dir"), -1, 1))
    return body + bytes([crc8(body)])


def decode_i2c(packet):
    """Validate and decode a guidance packet; invalid input raises ValueError."""
    if len(packet) != 20 or packet[:2] != b"\xd7\x02" or crc8(packet[:-1]) != packet[-1]:
        raise ValueError("Invalid guidance packet")
    _, _, seq, cmd, phase, label, flags, yaw, elev, travel, age, lost, direction = struct.unpack("<BBHBBBBhhHHHb", packet[:-1])
    if cmd >= len(COMMANDS) or phase >= len(PHASES) or (label != 255 and label >= len(LABELS)):
        raise ValueError("Invalid guidance enum")
    return dict(seq=seq, cmd=COMMANDS[cmd], phase=PHASES[phase],
                label=None if label == 255 else LABELS[label], flags=flags,
                fan=bool(flags & 1), goal_color="yellow" if flags & 2 else "orange",
                follow=bool(flags & 8), yaw_deg=yaw/100, elev_m=elev/100,
                travel_m=None if travel == 65535 else travel/100,
                age_ms=age, lost_ms=lost, search_dir=direction)
