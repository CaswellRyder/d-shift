"""Per-frame color tracker, mission phases and the Pi -> ESP32 message.

Guidance only: the ESP32 owns flight control, arena limits and failsafes.
Full-search results arrive late (about 0.5 s on the Pi); the tracker matches
them against where each track was when that frame was captured.
"""

from collections import deque
from dataclasses import dataclass, field
import json

import cv2
import numpy as np

from .goal_square import GOAL_COLORS, SQUARE_LABELS
from .vision import GREEN_PURPLE_HSV, HSV_RANGES

BALLOON_LABELS = ("green_balloon", "purple_balloon")
TRACK_HSV = {
    "green_balloon": GREEN_PURPLE_HSV[:1],
    "purple_balloon": GREEN_PURPLE_HSV[1:],
    # Same bands the goal search uses; orange also wraps through red hues.
    "orange_square": (HSV_RANGES["goal"][0], ((170, 45, 30), (179, 255, 255))),
    "yellow_square": (HSV_RANGES["goal"][1],),
}
PHASE_LABELS = {"balloon": BALLOON_LABELS, "capture": BALLOON_LABELS,
                "goal": tuple(SQUARE_LABELS.values())}
COMMANDS = ("SEARCH", "TRACK", "LOST", "FAN_ON", "ENTER", "AVOID")
PROTOCOL_VERSION = 1


def iou(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    inter = max(0.0, w) * max(0.0, h)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def center(box):
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


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

    def box_at(self, when):
        """Where the track was at `when`; the oldest kept box if `when` predates it."""
        best = self.box
        for stamp, box in reversed(self.history):
            best = box
            if stamp <= when:
                break
        return best


class ColorTracker:
    def __init__(self, width=320, height=240, min_pixels=4, max_misses=4, confirm_timeout=1.5,
                 window_scale=2.5, min_window=24, max_tracks=6, alpha=0.5, max_growth=1.6):
        self.width, self.height = width, height
        self.min_pixels, self.max_misses = min_pixels, max_misses
        self.confirm_timeout = confirm_timeout
        self.window_scale, self.min_window = window_scale, min_window
        self.max_tracks, self.alpha = max_tracks, alpha
        # Per-hit size change limit: real approach grows a target slowly at 20 fps, so a
        # sudden jump means the window latched onto other same-colored clutter.
        self.max_growth = max_growth
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

    def _hit(self, hsv, label, window, predicted, size):
        """Box of the label-colored blob nearest the predicted center, or None."""
        x1, y1, x2, y2 = window
        if x2 - x1 < 2 or y2 - y1 < 2:
            return None
        patch = hsv[y1:y2, x1:x2]
        mask = np.zeros(patch.shape[:2], np.uint8)
        for lo, hi in TRACK_HSV[label]:
            mask |= cv2.inRange(patch, lo, hi)
        count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        best, best_distance = None, None
        for index in range(1, count):
            if stats[index, cv2.CC_STAT_AREA] < self.min_pixels:
                continue
            side = max(stats[index, cv2.CC_STAT_WIDTH], stats[index, cv2.CC_STAT_HEIGHT])
            if not size / self.max_growth <= side <= size * self.max_growth + 2:
                continue
            distance = np.hypot(centroids[index][0] + x1 - predicted[0],
                                centroids[index][1] + y1 - predicted[1])
            if best is None or distance < best_distance:
                best, best_distance = index, distance
        if best is None:
            return None
        bx, by, bw, bh = stats[best, :4]
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
            box = self._hit(hsv, track.label, window, predicted, size)
            if box is None:
                track.misses += 1
                dx, dy = track.velocity[0] * dt, track.velocity[1] * dt
                track.box = [track.box[0] + dx, track.box[1] + dy, track.box[2] + dx, track.box[3] + dy]
            else:
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
            best, best_iou = None, 0.0
            for track in self.tracks:
                if track.label != det["label"] or track.id in claimed:
                    continue
                then = track.box_at(frame_time)
                overlap = iou(then, det["box"])
                if overlap <= 0:
                    # Small or fast targets may not overlap; accept a near center.
                    (tx, ty), (dx, dy) = center(then), center(det["box"])
                    size = max(det["box"][2] - det["box"][0], det["box"][3] - det["box"][1], 4)
                    overlap = 0.01 if np.hypot(tx - dx, ty - dy) <= size else 0.0
                if overlap > best_iou:
                    best, best_iou = track, overlap
            if best is None:
                unmatched.append(det)
                continue
            claimed.add(best.id)
            best.confirmed = max(best.confirmed, frame_time)
            best.score = float(det.get("score", best.score))
        for det in unmatched:
            if len(self.tracks) >= self.max_tracks:
                break
            box = [float(v) for v in det["box"]]
            # Start where the object was; widened windows re-find it in the next frames.
            track = Track(self.next_id, det["label"], box, confirmed=frame_time, seen=frame_time,
                          score=float(det.get("score", 0.0)), misses=1)
            track.history.append((frame_time, list(box)))
            self.tracks.append(track)
            self.next_id += 1
        self._drop_duplicates()
        return self.tracks

    def _drop_duplicates(self):
        kept = []
        for track in sorted(self.tracks, key=lambda t: (t.misses, -t.confirmed)):
            if all(t.label != track.label or iou(t.box, track.box) < 0.5 for t in kept):
                kept.append(track)
        self.tracks = sorted(kept, key=lambda t: t.id)


@dataclass
class MissionConfig:
    fresh_s: float = 0.8  # FAN_ON / ENTER need a full-search confirmation this recent.
    lost_s: float = 1.0  # Send LOST (hold) this long after losing the target, then SEARCH.
    fan_size: float = 0.30  # Balloon width / frame width to trigger FAN_ON (provisional).
    enter_size: float = 0.45  # Goal width / frame width to allow ENTER (provisional).
    centered: float = 0.25  # |x| and |y| limit for FAN_ON / ENTER.
    capture_timeout_s: float = 4.0  # No `captured` event in time: back to balloon.


class Mission:
    """balloon -> capture (FAN_ON) -> goal -> scored -> balloon. Emits one message per frame."""

    def __init__(self, goal_color, width=320, height=240, config=None):
        self.config = config or MissionConfig()
        self.width, self.height = width, height
        self.set_goal_color(goal_color)
        self.phase, self.fan, self.seq = "balloon", False, 0
        self.last_target_at, self.capture_started = None, None

    def set_goal_color(self, color):
        if color not in GOAL_COLORS:
            raise ValueError(f"Goal color must be one of {GOAL_COLORS}")
        self.goal_color = color

    @property
    def search_phase(self):
        return self.phase

    def _enter(self, phase):
        self.phase, self.last_target_at = phase, None

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
        return dict(label=track.label,
                    x=round(2 * cx / self.width - 1, 3), y=round(2 * cy / self.height - 1, 3),
                    size=round((x2 - x1) / self.width, 3),
                    age_ms=int(max(0.0, now - track.confirmed) * 1000),
                    seen_ms=int(max(0.0, now - track.seen) * 1000))

    def _ready(self, target, size_limit):
        c = self.config
        return (target["age_ms"] <= c.fresh_s * 1000 and target["size"] >= size_limit
                and abs(target["x"]) <= c.centered and abs(target["y"]) <= c.centered)

    def _best(self, tracks, labels, now):
        found = [t for t in tracks if t.label in labels and t.misses == 0]
        if not found:
            return None
        # Biggest (closest) first, then most recently confirmed.
        return max(found, key=lambda t: (t.box[2] - t.box[0], t.confirmed))

    def step(self, tracks, now):
        c = self.config
        cmd, target = None, None
        if self.phase in ("balloon", "capture"):
            track = self._best(tracks, BALLOON_LABELS, now)
            target = self.describe(track, now) if track else None
            if self.phase == "capture":
                cmd = "FAN_ON"
                if now - self.capture_started > c.capture_timeout_s:
                    self.capture_failed()
                    cmd = None
            elif target and self._ready(target, c.fan_size):
                self.fan, self.capture_started, cmd = True, now, "FAN_ON"
                self._enter("capture")
        else:
            ours = self._best(tracks, (SQUARE_LABELS[self.goal_color],), now)
            if ours:
                target = self.describe(ours, now)
                cmd = "ENTER" if self._ready(target, c.enter_size) else "TRACK"
            else:
                other = [t for t in tracks if t.label in SQUARE_LABELS.values()
                         and t.label != SQUARE_LABELS[self.goal_color] and t.misses == 0]
                if other:
                    target, cmd = self.describe(max(other, key=lambda t: t.box[2] - t.box[0]), now), "AVOID"
        if cmd is None:
            if target:
                cmd = "TRACK"
            elif self.last_target_at is not None and now - self.last_target_at <= c.lost_s:
                cmd = "LOST"
            else:
                cmd = "SEARCH"
        if target and cmd != "AVOID":
            self.last_target_at = now
        self.seq += 1
        return dict(v=PROTOCOL_VERSION, seq=self.seq, t_ms=int(now * 1000), phase=self.phase,
                    goal_color=self.goal_color, cmd=cmd, fan=self.fan, target=target)


def encode(message):
    """One compact JSON line, the wire format for the ESP32 link."""
    if message["cmd"] not in COMMANDS:
        raise ValueError("Unknown command")
    return json.dumps(message, separators=(",", ":")) + "\n"
