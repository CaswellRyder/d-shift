"""The frame step shared by the flight runner and the lab recorder.

No camera, encoding, storage, HTTP, or actuator transport belongs in this step.
"""

from .flight import Mission


class FlightPipeline:
    def __init__(self, worker, tracker, mission, telemetry=None):
        self.worker, self.tracker, self.mission = worker, tracker, mission
        self.telemetry = telemetry
        self.phase_since = float('-inf')

    def configure(self, phase, goal_color, now):
        if phase not in ('balloon', 'goal'):
            raise ValueError('Choose balloon or goal')
        # Keep the launch configuration (custom sizes and gates) across phase/colour changes.
        mission = Mission(goal_color, self.tracker.width, self.tracker.height,
                          config=self.mission.config, target_sizes=self.mission.target_sizes, vector=self.mission.vector)
        if phase == 'goal':
            mission.captured()
        self.mission = mission
        self.tracker.clear()
        self.phase_since = now

    def step(self, rgb, now):
        self.tracker.update(rgb, now)
        result = self.worker.poll()
        if (result is not None and result['phase'] == self.mission.search_phase
                and self.phase_since <= result['frame_time'] <= now):
            self.tracker.merge(result['detections'], result['frame_time'], now)
        tracks = [dict(id=t.id, label=t.label, box=list(t.box), score=float(t.score),
                       center_h=t.center_h, confirmed=float(t.confirmed), seen=float(t.seen), misses=t.misses)
                  for t in self.tracker.tracks]
        phase = self.mission.phase
        message = self.mission.step(self.tracker.tracks, now)
        if self.mission.phase != phase:
            self.tracker.clear()
            self.phase_since = now
        submitted = self.worker.submit(rgb, now, self.mission.search_phase)
        row = dict(message=message, tracks=tracks, search_result=result,
                   submitted=submitted, phase_before=phase,
                   selected_track_id=getattr(self.mission, 'selected_track_id', None))
        if self.telemetry is not None:
            row['flight_telemetry'] = self.telemetry.observe(row, now)
        return row
