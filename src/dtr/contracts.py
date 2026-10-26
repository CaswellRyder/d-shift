"""Version 1 module contracts. Coordinates and ranges are meters; angles degrees.

frame_time is capture time in monotonic seconds. age_ms is time since full-search
confirmation; seen_ms is time since the last color hit, both relative to frame_time.
"""

from dataclasses import asdict, dataclass
from typing import Literal

CONTRACT_VERSION = 1


@dataclass(frozen=True)
class Target:
    label: str
    score: float
    box_px: tuple[float, float, float, float]
    bearing_deg: float
    elevation_deg: float
    cam_xyz_m: tuple[float, float, float]
    body_xyz_m: tuple[float, float, float]  # forward, right, up
    range_m: float  # Distance from body origin; camera range is norm(cam_xyz_m).
    range_err_m: float
    clipped: bool
    yaw_err_deg: float
    climb_err_deg: float
    size_calibrated: bool
    lens_calibrated: bool
    age_ms: float
    seen_ms: float
    range_is_upper_bound: bool = False

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class TargetReport:
    frame_time: float
    phase: Literal["balloon", "goal"]
    target: Target | None
    avoid: Target | None
    ready: bool

    def to_dict(self):
        return dict(contract_version=CONTRACT_VERSION, **asdict(self))
