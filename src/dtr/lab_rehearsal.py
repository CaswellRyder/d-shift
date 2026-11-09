"""Operator rehearsal records. No controller transport or motor authority.

Receipt, application to this rehearsal state, and physical application are distinct.
Possession is only an operator observation, never inferred from visual target loss.
"""

from collections import deque
from copy import deepcopy
import math
import uuid


ACTIONS = frozenset(('vacuum_intake', 'vacuum_off', 'vacuum_reverse',
                     'capture_observed', 'release_observed', 'capture_failed', 'reset_trial'))
TRANSITIONS = {'capture_observed': 'goal', 'release_observed': 'balloon',
               'capture_failed': 'balloon', 'reset_trial': 'balloon'}
RECORD_ONLY = dict(bench_only=True, transmitted=False, applied=False, controller_ack=None)


def validate_event(payload):
    if (not isinstance(payload, dict) or not isinstance(payload.get('action'), str)
            or payload['action'] not in ACTIONS):
        raise ValueError('Choose a supported rehearsal action')
    if set(payload) - {'action', 'duration_s', 'note'}:
        raise ValueError('Unknown rehearsal settings')
    action, note = payload['action'], payload.get('note', '')
    if not isinstance(note, str) or len(note) > 500:
        raise ValueError('Rehearsal note is limited to 500 characters')
    timed = action in ('vacuum_intake', 'vacuum_reverse')
    if not timed and 'duration_s' in payload:
        raise ValueError('Duration is only valid for intake or reverse requests')
    duration = payload.get('duration_s', 1.) if timed else None
    if timed and (isinstance(duration, bool) or not isinstance(duration, (int, float))
                  or not math.isfinite(duration) or not .1 <= duration <= 5):
        raise ValueError('Vacuum request duration must be 0.1 to 5 seconds')
    return dict(action=action, note=note, duration_s=float(duration) if timed else None)


class Rehearsal:
    """Bounded queued events; caller serializes receive/apply/snapshot with its lock."""

    def __init__(self, record, max_pending=64):
        self.record = record
        self.server_session = uuid.uuid4().hex
        self.pending = deque()
        self.recent = deque(maxlen=64)
        self.max_pending = max_pending
        self.sequence = 0
        self.persistence_error = False
        self.vacuum = dict(mode='off', expires_at_monotonic=None, event_id=None)
        self.possession = dict(state='unknown', source='unavailable', event_id=None)

    def _save(self, event):
        self.record(deepcopy(event))  # Failure must reach the caller before success.
        self.recent.append(deepcopy(event))

    def _terminal(self, event):
        """A completed state change cannot become pending again after a log error."""
        try:
            self._save(event)
        except Exception:
            self.persistence_error = True
            self.recent.append(dict(deepcopy(event), persistence='failed'))
            raise

    def receive(self, payload, now, *, frame_sequence, frame_time):
        settings = validate_event(payload)
        if len(self.pending) >= self.max_pending and settings['action'] != 'vacuum_off':
            raise RuntimeError('Rehearsal event queue is full; wait for the next frame')
        event = dict(settings, event_id=f'{self.server_session}:{self.sequence + 1}',
                     server_session=self.server_session, received_at_monotonic=float(now),
                     frame_sequence=frame_sequence, frame_time=frame_time,
                     source='operator_observation' if settings['action'] in
                     ('capture_observed', 'release_observed', 'capture_failed') else 'operator_request',
                     status='received', **RECORD_ONLY)
        self.sequence += 1
        self._save(event)
        if settings['action'] == 'vacuum_off':
            # Off takes effect even when the camera stalls. Older queued vacuum
            # requests are superseded, so a later frame cannot revive them.
            self.vacuum = dict(mode='off', expires_at_monotonic=None, event_id=event['event_id'])
            for queued in self.pending:
                if queued['action'].startswith('vacuum_'):
                    queued['superseded'] = True
            applied = dict(event, status='applied_to_rehearsal', applied_at_monotonic=float(now))
            self._terminal(applied)
        else:
            self.pending.append(event)
        return deepcopy(event)

    def apply(self, now, transition):
        """Apply at a camera-frame boundary; keep pending evidence on failure."""
        applied_ids = []
        while self.pending:
            event = self.pending[0]
            if event['received_at_monotonic'] > now:
                break  # Never attach an operator event to an earlier camera frame.
            action = event['action']
            result = dict(event, status='applied_to_rehearsal', applied_at_monotonic=float(now))
            if event.get('superseded'):
                result['status'] = 'superseded'
            elif action.startswith('vacuum_'):
                expiry = event['received_at_monotonic'] + event['duration_s']
                result['expires_at_monotonic'] = expiry
                if now >= expiry:
                    result['status'] = 'expired_before_application'
                    self.vacuum = dict(mode='off', expires_at_monotonic=expiry,
                                       event_id=event['event_id'])
                else:
                    self.vacuum = dict(mode=action.removeprefix('vacuum_'),
                                       expires_at_monotonic=expiry, event_id=event['event_id'])
            else:
                transition(TRANSITIONS[action], now)
                if action == 'capture_observed':
                    self.possession = dict(state='held', source='manual_operator', event_id=event['event_id'])
                else:
                    self.vacuum = dict(mode='off', expires_at_monotonic=None, event_id=event['event_id'])
                    self.possession = dict(state='released' if action == 'release_observed' else 'unknown',
                                           source='manual_operator' if action != 'reset_trial' else 'unavailable',
                                           event_id=event['event_id'])
            self.pending.popleft()
            self._terminal(result)
            applied_ids.append(event['event_id'])
        return applied_ids

    def cancel_pending(self, now):
        self.vacuum = dict(mode='off', expires_at_monotonic=None, event_id=None)
        while self.pending:
            event = self.pending.popleft()
            self._terminal(dict(event, status='not_applied_loop_stopped',
                                applied_at_monotonic=float(now)))

    def snapshot(self, now, row=None, *, source=None, frame_sequence=None):
        vacuum = deepcopy(self.vacuum)
        if vacuum['expires_at_monotonic'] is not None and now >= vacuum['expires_at_monotonic']:
            vacuum['mode'] = 'off'
        model = None
        if row is not None:
            message = row['message']
            outgoing = row.get('flight_telemetry', {}).get('outgoing', {})
            model = dict(command=message['cmd'], fan_requested=message['fan'],
                         selected_track_id=row.get('selected_track_id'),
                         target=deepcopy(message.get('target')),
                         guidance=deepcopy(outgoing.get('draft_packet')))
        return dict(schema='lab-rehearsal-v1', server_session=self.server_session,
                    source=source, frame_sequence=frame_sequence,
                    operator_vacuum=vacuum, possession=deepcopy(self.possession),
                    model_request=model, persistence_error=self.persistence_error, **RECORD_ONLY)
