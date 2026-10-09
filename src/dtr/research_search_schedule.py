"""Research-only full/bright search dispatch; not a flight-control deadline."""


class SearchSchedule:
    """Dispatch full search after 500 ms or at most two completed bright frames.

    Time is measured from the previous successful full search's START. A running
    frame cannot be preempted: lateness is measured, not hidden as a hard bound.
    This carries no detections/tracks and never commands hardware.
    """

    interval_ns = 500_000_000
    max_bright_frames = 2

    def __init__(self):
        self.last_full_start = None
        self.bright_frames = 0
        self.last_time = None
        self.pending = None

    def _time(self, now):
        if type(now) is not int or now < 0 or (self.last_time is not None and now < self.last_time):
            raise ValueError("Require nonnegative monotonic integer nanoseconds")
        self.last_time = now

    def begin(self, now, force_full=False):
        if self.pending is not None:
            raise ValueError("Previous search must finish before another begins")
        if type(force_full) is not bool:
            raise ValueError("force_full must be boolean")
        self._time(now)
        age = None if self.last_full_start is None else now - self.last_full_start
        if force_full:
            reason = "forced_full"
        elif age is None:
            reason = "startup_or_failure"
        elif age >= self.interval_ns:
            reason = "elapsed_budget"
        elif self.bright_frames >= self.max_bright_frames:
            reason = "frame_cap"
        else:
            reason = "bright_between_full"
        mode = "mser_bright" if reason == "bright_between_full" else "mser_direct"
        decision = dict(
            mode=mode,
            reason=reason,
            full_age_ns=age,
            due_lateness_ns=max(0, age - self.interval_ns) if age is not None else 0,
        )
        self.pending = (now, mode)
        return decision

    def finish(self, now, success=True):
        if self.pending is None:
            raise ValueError("No pending search")
        if type(success) is not bool:
            raise ValueError("success must be boolean")
        self._time(now)
        start, mode = self.pending
        if not success:
            self.last_full_start, self.bright_frames = None, 0
        elif mode == "mser_direct":
            self.last_full_start, self.bright_frames = start, 0
        else:
            self.bright_frames += 1
        self.pending = None
