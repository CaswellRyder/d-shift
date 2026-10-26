"""Sparse fisheye geometry on already-flipped runtime images; no image remapping."""

import json
from pathlib import Path

import cv2
import numpy as np


class LensModel:
    def __init__(self, K, D, image_size, calibrated=False, rms=None):
        self.K = np.array(K, dtype=np.float64, copy=True)
        self.D = np.asarray(D, dtype=np.float64).reshape(4, 1).copy()
        self.width, self.height = map(int, image_size)
        self.calibrated, self.rms = bool(calibrated), rms
        if (self.K.shape != (3, 3) or not np.isfinite(self.K).all()
                or not np.isfinite(self.D).all() or min(self.width, self.height) <= 0
                or min(self.K[0, 0], self.K[1, 1]) <= 0):
            raise ValueError("Invalid fisheye calibration")

    @classmethod
    def from_calibration(cls, path, width=None, height=None):
        """Load calibration for flipped frames, optionally scaling to runtime dimensions."""
        data = json.loads(Path(path).read_text())
        if data.get("flipped") is not True:
            raise ValueError("Calibration must describe flipped runtime images")
        lens = cls(data["K"], data["D"], data["image_size"], True, data["rms"])
        if (width is None) != (height is None):
            raise ValueError("Supply both runtime width and height")
        return lens if width is None else lens.scaled(width, height)

    @classmethod
    def ideal_equidistant(cls, width=320, height=240, fov_deg=120.0, axis="diagonal"):
        """Uncalibrated fallback: equidistant fisheye, square pixels, no crop.

        Our lens is 120 degrees corner to corner (about 96 wide, 72 tall at 4:3; we
        measured ~95 side to side). `axis="horizontal"` reads fov_deg edge to edge instead.
        """
        if not 0 < fov_deg < 180:
            raise ValueError("FOV must be between 0 and 180 degrees")
        half = {"diagonal": np.hypot(width, height) / 2, "horizontal": width / 2}.get(axis)
        if half is None:
            raise ValueError("axis must be 'diagonal' or 'horizontal'")
        f = half / np.radians(fov_deg / 2)
        return cls([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]],
                   np.zeros(4), (width, height))

    def scaled(self, width, height):
        K = self.K.copy()
        K[0] *= width / self.width
        K[1] *= height / self.height
        return LensModel(K, self.D, (width, height), self.calibrated, self.rms)

    def rays(self, points_px):
        """Return N unit rays (x right, y down, z forward). Reject failed inversions.

        OpenCV's inverse uses normalized pinhole coordinates, valid below 90 degrees
        off axis. Its failure sentinel is finite, so explicitly reject it too.
        """
        points = np.asarray(points_px, np.float64).reshape(-1, 1, 2)
        if not np.isfinite(points).all():
            raise ValueError("Non-finite pixels")
        if not len(points):
            return np.empty((0, 3))
        uv = cv2.fisheye.undistortPoints(
            points, self.K, self.D,
            criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 40, 1e-12)
        ).reshape(-1, 2)
        if not np.isfinite(uv).all() or np.any(np.abs(uv) >= 1e6):
            raise ValueError("Fisheye inverse failed or ray approaches 90 degrees")
        rays = np.column_stack((uv, np.ones(len(uv))))
        return rays / np.linalg.norm(rays, axis=1, keepdims=True)

    def in_view(self, points_px, exponent=4.0, scale=1.0):
        """True inside the trusted "squoval" |u|^n + |v|^n <= scale^n (u, v: -1..1 from center).

        Keeps the full edges at frame mid-lines but cuts the warped far corners;
        exponent 4 drops about 7% of the frame. Tune after calibration.
        """
        points = np.asarray(points_px, np.float64).reshape(-1, 2)
        u = (points[:, 0] - self.width / 2) / (self.width / 2)
        v = (points[:, 1] - self.height / 2) / (self.height / 2)
        return np.abs(u) ** exponent + np.abs(v) ** exponent <= scale ** exponent

    def grid(self, step=1):
        """Return (bearing, elevation) degree arrays of shape (ceil(h/step), ceil(w/step))."""
        if not isinstance(step, int) or step < 1:
            raise ValueError("step must be a positive integer")
        x, y = np.meshgrid(np.arange(0, self.width, step), np.arange(0, self.height, step))
        rays = self.rays(np.column_stack((x.ravel(), y.ravel()))).reshape(*x.shape, 3)
        return (np.degrees(np.arctan2(rays[..., 0], rays[..., 2])),
                np.degrees(np.arctan2(-rays[..., 1],
                                     np.hypot(rays[..., 0], rays[..., 2]))))
