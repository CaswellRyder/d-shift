"""Detection/tracking and geometry only. No mission gates or actuator decisions."""


import numpy as np

from .contracts import Target, TargetReport
from .flight import BALLOON_LABELS, ColorTracker
from .goal_square import GOAL_COLORS, GOAL_LABELS, LABELS_BY_COLOR
from .lens import LensModel
from .search_worker import SearchWorker
from .target_geometry import TargetGeometry, load_target_sizes, relative_range, steering


def best_track(tracks, labels, sizes=None):
    """Nearest goal (or widest balloon), then newest confirmation, with a colour hit."""
    if sizes is not None:
        return min((t for t in tracks if t.label in labels and t.misses == 0),
                   key=lambda t: (relative_range(t.label, t.box, sizes), -t.confirmed), default=None)
    return max((t for t in tracks if t.label in labels and t.misses == 0),
               key=lambda t: (t.box[2] - t.box[0], t.confirmed), default=None)


class FindTarget:
    def __init__(self, factory=None, *, worker=None, tracker=None, lens=None,
                 geometry=None, width=320, height=240, view_exponent=4.0, view_scale=1.0):
        """Supply a picklable search factory (e.g. ModelSearch), or an owned worker.

        Configuration is loaded here, never in the per-frame calls. ready means
        models have warmed up, including while the worker is processing a frame.
        """
        if worker is None and factory is None:
            raise ValueError("Supply a search factory or worker")
        self.tracker = tracker or ColorTracker(width, height)
        self.lens = (geometry.lens if geometry is not None else
                     (lens or LensModel.ideal_equidistant(width, height)).scaled(width, height))
        if (self.lens.width, self.lens.height) != (width, height):
            raise ValueError("Geometry dimensions must match runtime frames")
        self.geometry = geometry or TargetGeometry(self.lens)
        self.target_sizes = getattr(self.geometry, "sizes", None)
        if self.target_sizes is None:
            self.target_sizes = load_target_sizes()
        self.worker = worker if worker is not None else SearchWorker(
            factory, shape=(height, width, 3))
        self.phase, self.phase_since = None, None
        self.geometry_failures = 0
        self.view_exponent, self.view_scale = float(view_exponent), float(view_scale)

    def start(self):
        self.worker.start()
        return self

    def wait_ready(self, timeout=60.0):
        self.worker.wait_ready(timeout)
        return self

    def close(self):
        self.worker.close()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.close()

    def balloon(self, rgb, frame_time):
        return self._frame(rgb, frame_time, "balloon", BALLOON_LABELS)

    def goal(self, rgb, frame_time, target_color):
        if target_color not in GOAL_COLORS:
            raise ValueError(f"Unknown goal color: {target_color}")
        return self._frame(rgb, frame_time, "goal", LABELS_BY_COLOR[target_color])

    def _frame(self, rgb, frame_time, phase, labels):
        if phase != self.phase:
            self.tracker.clear()
            self.phase, self.phase_since = phase, frame_time
        self.tracker.update(rgb, frame_time)
        result = self.worker.poll()
        # Timestamp also prevents balloon -> goal -> balloon from accepting an old balloon result.
        if (result is not None and result["phase"] == phase
                and self.phase_since <= result["frame_time"] <= frame_time):
            self.tracker.merge(result["detections"], result["frame_time"], frame_time)
        # Far corners are untrusted lens geometry: keep tracking there, but don't report.
        tracks = [t for t in self.tracker.tracks
                  if self.lens.in_view([((t.box[0] + t.box[2]) / 2, (t.box[1] + t.box[3]) / 2)],
                                       self.view_exponent, self.view_scale)[0]]
        sizes = self.target_sizes if phase == "goal" else None
        target = best_track(tracks, labels, sizes)
        avoid = (best_track(tracks,
                            tuple(v for v in GOAL_LABELS if v not in labels), sizes)
                 if phase == "goal" else None)
        ready = self.worker.ready or getattr(self.worker, "state", None) == "busy"
        self.worker.submit(rgb, frame_time, phase)
        return TargetReport(float(frame_time), phase, self._locate(target, frame_time),
                            self._locate(avoid, frame_time), bool(ready))

    def _locate(self, track, frame_time):
        """Geometry for one track; a lens-inversion failure drops that target, not the loop."""
        if track is None:
            return None
        try:
            return self.geometry.from_track(track, frame_time)
        except ValueError:
            self.geometry_failures += 1
            return None


class FakeFindTarget:
    """Replay absolute-time keyframes [(seconds, (forward,right,up) | None), ...].

    Optional avoid_keyframes use the same format. Interpolation requires two present
    endpoints; a None endpoint makes the open interval a gap. Outside the script is
    None. Exact non-None keyframes are visible, including a lone keyframe.
    Targets are scripted ground truth, not calibrated physical measurements.
    """

    def __init__(self, keyframes, avoid_keyframes=()):
        self.keyframes = self._validate(keyframes)
        self.avoid_keyframes = self._validate(avoid_keyframes)
        self.ready = False

    @staticmethod
    def _validate(frames):
        frames = [(float(t), None if p is None else tuple(map(float, p))) for t, p in frames]
        if (any(not np.isfinite(t) or (p is not None and
                (len(p) != 3 or not np.isfinite(p).all() or np.linalg.norm(p) == 0))
                for t, p in frames)
                or any(a[0] >= b[0] for a, b in zip(frames, frames[1:]))):
            raise ValueError("Keyframes need increasing finite times and nonzero 3D vectors")
        return frames

    def start(self):
        self.ready = True
        return self

    def wait_ready(self, timeout=60.0):
        return self.start()

    def close(self):
        self.ready = False

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.close()

    @staticmethod
    def _sample(frames, when, label):
        point = None
        for t, p in frames:
            if when == t:
                point = p
                break
        else:
            for (t0, p0), (t1, p1) in zip(frames, frames[1:]):
                if t0 < when < t1 and p0 is not None and p1 is not None:
                    point = np.array(p0) + (np.array(p1) - p0) * ((when - t0) / (t1 - t0))
                    break
        if point is None or np.linalg.norm(point) == 0:
            return None
        forward, right, up = map(float, point)
        yaw, climb = steering(point)
        return Target(label, 1.0, (0., 0., 0., 0.), yaw, climb, (right, -up, forward),
                      (forward, right, up), float(np.linalg.norm(point)), 0., False,
                      yaw, climb, False, False, 0., 0.)

    def balloon(self, rgb, frame_time):
        return TargetReport(float(frame_time), "balloon",
                            self._sample(self.keyframes, frame_time, "green_balloon"),
                            None, self.ready)

    def goal(self, rgb, frame_time, target_color):
        if target_color not in GOAL_COLORS:
            raise ValueError(f"Unknown goal color: {target_color}")
        # Scripted XYZ has no observed silhouette; retain a square fixture label.
        target = self._sample(self.keyframes, frame_time, f"{target_color}_square")
        other = next(f"{color}_square" for color in GOAL_COLORS if color != target_color)
        avoid = self._sample(self.avoid_keyframes, frame_time, other)
        return TargetReport(float(frame_time), "goal", target, avoid, self.ready)
