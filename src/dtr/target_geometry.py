"""Angular box sizing, assuming a target facing the camera (goal pose is unknown).

Body coordinates: x forward, y right, z up. Positive mount yaw turns right,
positive pitch raises the optical axis, positive roll lowers the right side.
Rotation order is yaw @ pitch @ roll on converted camera axes; translation is
camera position in body meters. A clipped range is an upper bound only under
these size/pose assumptions. Spheres use the exact tangent-cone range. Flat goals
use the smaller width/height range to tolerate yaw. Image roll widens boxes and reads
closer (about 8% at 5 degrees of roll).
"""

import json
from pathlib import Path

import numpy as np

from .contracts import Target

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def load_target_sizes(path=None):
    sizes = json.loads(Path(path or CONFIGS / "target_sizes.json").read_text())
    for size in sizes.values():
        if (not np.isfinite(size["width_m"]) or size["width_m"] <= 0
                or size["shape"] not in ("round", "square", "circle", "triangle")):
            raise ValueError("Invalid target size")
        if size["shape"] != "round":
            if (not np.isfinite(size.get("height_m", 0)) or size.get("height_m", 0) <= 0):
                raise ValueError("Flat targets require a positive height_m")
    return sizes


def relative_range(label, box, sizes):
    """Physical extent / pixels; smaller is nearer, using the unforeshortened axis."""
    size = sizes[label]
    w, h = box[2] - box[0], box[3] - box[1]
    if w <= 0 or h <= 0:
        return float("inf")
    return min(size["width_m"] / w, size["height_m"] / h)


def steering(body):
    forward, right, up = body
    return (float(np.degrees(np.arctan2(right, forward))),
            float(np.degrees(np.arctan2(up, np.hypot(forward, right)))))


class CameraMount:
    def __init__(self, yaw_deg=0, pitch_deg=0, roll_deg=0, translation_m=(0, 0, 0), calibrated=False):
        self.calibrated = bool(calibrated)
        y, p, r = np.radians([yaw_deg, pitch_deg, roll_deg])
        cy, sy, cp, sp, cr, sr = np.cos(y), np.sin(y), np.cos(p), np.sin(p), np.cos(r), np.sin(r)
        yaw = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        pitch = np.array([[cp, 0, -sp], [0, 1, 0], [sp, 0, cp]])
        roll = np.array([[1, 0, 0], [0, cr, sr], [0, -sr, cr]])
        self.rotation = yaw @ pitch @ roll
        self.translation = np.asarray(translation_m, float)
        if (self.translation.shape != (3,) or not np.isfinite(self.translation).all()
                or not np.isfinite(self.rotation).all()):
            raise ValueError("Invalid camera mount")

    @classmethod
    def from_config(cls, path):
        return cls(**json.loads(Path(path).read_text()))

    def transform(self, cam):
        return self.rotation @ np.array([cam[2], cam[0], -cam[1]]) + self.translation


class TargetGeometry:
    def __init__(self, lens, sizes_path=None, mount=None, pixel_sigma=1.0):
        self.lens = lens
        self.sizes = load_target_sizes(sizes_path)
        self.mount = mount or CameraMount.from_config(CONFIGS / "camera_mount.json")
        self.pixel_sigma = float(pixel_sigma)
        if not np.isfinite(self.pixel_sigma) or self.pixel_sigma < 0:
            raise ValueError("pixel_sigma must be finite and nonnegative")

    def target(self, label, score, box_px, age_ms=0.0, seen_ms=0.0):
        box = np.asarray(box_px, float)
        if box.shape != (4,) or not np.isfinite(box).all():
            raise ValueError("Invalid box")
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Box must have positive extent")
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        # One vectorized inverse for center, nominal extent, and both error bounds.
        points = [[cx, cy]]
        for delta in (0, self.pixel_sigma, -self.pixel_sigma):
            # Avoid crossing the two edges for subpixel/tiny boxes.
            dx = max(delta, -(x2 - x1) / 2 + 1e-6)
            dy = max(delta, -(y2 - y1) / 2 + 1e-6)
            points.extend([[x1 - dx, cy], [x2 + dx, cy],
                           [cx, y1 - dy], [cx, y2 + dy]])
        rays = self.lens.rays(points)
        size = self.sizes[label]
        ranges = []
        for i in (1, 5, 9):
            pairs = rays[i:i + 4].reshape(2, 2, 3)
            # atan2 is well conditioned even for very small extents.
            angles = np.arctan2(np.linalg.norm(np.cross(pairs[:, 0], pairs[:, 1]), axis=1),
                                np.sum(pairs[:, 0] * pairs[:, 1], axis=1))
            # Sphere: tangent rays meet the center at radius / sin. Flat face: half-width / tan.
            if size["shape"] == "round":
                ranges.append(size["width_m"] / 2 / np.sin(max(angles) / 2))
            else:
                ranges.append(min(np.array([size["width_m"], size["height_m"]])
                                  / 2 / np.tan(angles / 2)))
        distance = float(ranges[0])
        error = float(max(abs(ranges[1] - distance), abs(ranges[2] - distance)))
        cam = rays[0] * distance
        body = self.mount.transform(cam)
        yaw, climb = steering(body)
        bearing, elevation = steering((cam[2], cam[0], -cam[1]))
        clipped = bool(x1 <= 1 or y1 <= 1 or x2 >= self.lens.width - 1
                       or y2 >= self.lens.height - 1)
        return Target(label, float(score), tuple(map(float, box)), bearing, elevation,
                      tuple(map(float, cam)), tuple(map(float, body)),
                      float(np.linalg.norm(body)), error,
                      clipped, yaw, climb, bool(size["calibrated"]), self.lens.calibrated,
                      float(age_ms), float(seen_ms), clipped)

    def from_track(self, track, frame_time):
        return self.target(track.label, track.score, track.box,
                           max(0, frame_time - track.confirmed) * 1000,
                           max(0, frame_time - track.seen) * 1000)
