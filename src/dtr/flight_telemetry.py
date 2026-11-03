"""Camera coordinates and proposed ESP32 observations, with no actuator transport.

The existing v1 wire line is kept exact. The additional packet is a draft: yaw
and climb angles are geometric pointing errors, not servo angles, PWM, a world
position, or a tuned controller. Operator tape measurements are never inputs.
"""

import json
from pathlib import Path

import numpy as np

from .flight import PHASE_LABELS, encode, encode_v2
from .lens import LensModel
from .target_geometry import CONFIGS, CameraMount, TargetGeometry


def add_mapping_arguments(parser):
    parser.add_argument('--lens-calibration', help='Checkerboard fisheye calibration JSON')
    parser.add_argument('--camera-mount', default=str(CONFIGS / 'camera_mount.json'))
    parser.add_argument('--target-sizes', default=str(CONFIGS / 'target_sizes.json'))


def mapping_from_args(args):
    lens = (LensModel.from_calibration(args.lens_calibration, 320, 240)
            if args.lens_calibration else LensModel.ideal_equidistant())
    geometry = TargetGeometry(lens, sizes_path=args.target_sizes,
                              mount=CameraMount.from_config(args.camera_mount))
    return FlightTelemetry(geometry)


class FlightTelemetry:
    def __init__(self, geometry=None):
        self.geometry = geometry or TargetGeometry(LensModel.ideal_equidistant())
        self.lens = self.geometry.lens
        self.mapping_status = dict(
            model='calibrated_fisheye' if self.lens.calibrated else 'ideal_equidistant_estimate',
            lens_calibrated=bool(self.lens.calibrated),
            mount_calibrated=bool(getattr(self.geometry.mount, 'calibrated', False)),
            range_uncertainty='Pixel-edge error only; excludes size, pose and calibration error',
            metric_coordinates='estimates; not arena position',
        )
        self._mapping = self._build_mapping()

    def _build_mapping(self):
        samples = []
        # Sparse pixel grid mapped to angles; do not stretch or resample camera frames.
        for v in range(0, self.lens.height + 1, 20):
            for u in range(0, self.lens.width + 1, 20):
                trusted = bool(self.lens.in_view([[u, v]])[0])
                try:
                    ray = self.lens.rays([[u, v]])[0]
                    bearing = float(np.degrees(np.arctan2(ray[0], ray[2])))
                    elevation = float(np.degrees(np.arctan2(-ray[1], np.hypot(ray[0], ray[2]))))
                except ValueError:
                    bearing, elevation, trusted = None, None, False
                samples.append(dict(u=u, v=v, bearing_deg=bearing,
                                    elevation_deg=elevation, trusted=trusted))
        return dict(width=self.lens.width, height=self.lens.height, samples=samples,
                    optical_center_px=dict(u=float(self.lens.K[0, 2]), v=float(self.lens.K[1, 2])),
                    image_center_px=dict(u=self.lens.width / 2, v=self.lens.height / 2),
                    axes=dict(camera='right, down, forward (m)', body='forward, right, up (m)',
                              bearing='positive right (degrees)', elevation='positive up (degrees)'),
                    **self.mapping_status)

    def mapping(self):
        return self._mapping

    def _locate(self, track, now):
        x1, y1, x2, y2 = track['box']
        if not self.lens.in_view([[(x1 + x2) / 2, (y1 + y2) / 2]])[0]:
            raise ValueError('Target centre is outside the trusted lens region')
        target = self.geometry.target(track['label'], track['score'], track['box'],
                                      max(0., now - track['confirmed']) * 1000,
                                      max(0., now - track['seen']) * 1000)
        result = target.to_dict()
        result['mount_calibrated'] = self.mapping_status['mount_calibrated']
        result['goal_pose_ambiguous'] = track['label'] in PHASE_LABELS['goal']
        result['range_qualified'] = False  # No measured pose/size uncertainty model yet.
        result['age_ms'] = round(result['age_ms'], 6)
        result['seen_ms'] = round(result['seen_ms'], 6)
        json.dumps(result, allow_nan=False)  # Singular inversions must not poison a frame log.
        return result

    def observe(self, row, now):
        message = row['message']
        published = message.get('target')
        selected, status = None, 'no_target'
        if published is not None:
            tracks = [t for t in row['tracks'] if t['label'] == published['label']
                      and t['misses'] == 0]
            selected_id = row.get('selected_track_id')
            if selected_id is not None:
                tracks = [t for t in tracks if t['id'] == selected_id]
            else:
                # Replay support for older rows: bind to the published point, not a
                # second independent target choice.
                def matches(t):
                    x1, y1, x2, y2 = t['box']
                    return (round((x1 + x2) / self.lens.width - 1, 3) == published.get('x')
                            and round((y1 + y2) / self.lens.height - 1, 3) == published.get('y')
                            and round((x2 - x1) / self.lens.width, 3) == published.get('size'))
                tracks = [t for t in tracks if matches(t)]
            if tracks:
                track = tracks[0]
                try:
                    selected = self._locate(track, now)
                    status = 'estimated'
                except (ValueError, KeyError, FloatingPointError) as error:
                    status = 'geometry_unavailable: ' + str(error)
        avoiding = message['cmd'] == 'AVOID'
        target, avoid = (None, selected) if avoiding else (selected, None)
        blockers = ['esp32_control_contract_not_approved', 'imu_and_controller_telemetry_unavailable']
        if not self.mapping_status['lens_calibrated']:
            blockers.append('lens_uncalibrated')
        if not self.mapping_status['mount_calibrated']:
            blockers.append('camera_mount_unmeasured')
        if selected is None:
            blockers.append('no_valid_target_geometry')
        else:
            if not selected['size_calibrated']:
                blockers.append('target_size_unmeasured')
            if selected['clipped']:
                blockers.append('target_clipped')
            if selected['goal_pose_ambiguous']:
                blockers.append('goal_pose_range_ambiguous')
            if selected['age_ms'] > 800 or selected['seen_ms'] > 100:
                blockers.append('target_stale')
        pointing = (target is not None and target['age_ms'] <= 800
                    and target['seen_ms'] <= 100 and not target['clipped']
                    and message['cmd'] in ('TRACK', 'FAN_ON', 'ENTER'))
        # These errors describe where to look. ESP32 gains/limits/actuator mix are unknown.
        guidance = dict(yaw_error_deg=target['yaw_err_deg'] if pointing else None,
                        climb_error_deg=target['climb_err_deg'] if pointing else None,
                        forward_thrust=None)
        camera_delta, navigation_delta, image_error = None, None, None
        if target is not None:
            right, down, forward = target['cam_xyz_m']
            camera_delta = dict(x_right_m=right, y_up_m=-down, z_forward_m=forward)
            forward, right, up = target['body_xyz_m']
            navigation_delta = dict(x_right_m=right, y_up_m=up, z_forward_m=forward)
            image_error = dict(x_right=published['x'], y_up=-published['y'])
        published_vector = message.get('target') or {}
        packet = dict(target_vector=dict(**{k: published_vector.get(k) for k in ('yaw_deg', 'elev_m', 'travel_m')}, source='connor_vector_creation'), schema='pi-esp32-guidance-draft-v1', seq=message['seq'],
                      frame_time_s=float(now), timestamp_kind='post_acquisition_monotonic',
                      sensor_timestamp='recorded separately in the frame row; no IMU clock alignment yet',
                      target_goal_color=message['goal_color'],
                      body_axes='forward,right,up', position_units='m', angle_units='deg',
                      delta_axes='x_right,y_up,z_forward', coordinate_frame='vehicle_relative',
                      delta_meaning='current target offset; not a change between consecutive frames',
                      camera_delta_m=camera_delta, navigation_delta_m=navigation_delta,
                      camera_error_normalized=image_error,
                      image_error_origin='image centre; metric XYZ uses the calibrated optical axis and mount',
                      desired_standoff_m=None, forward_error_m=None,
                      intent=message['cmd'], target=target, avoid=avoid,
                      relative_guidance=guidance, world_heading_deg=None,
                      fan_request=bool(message['fan']), actuator_command=None,
                      control_ready=False, blockers=blockers,
                      mission_status='legacy mission; SHOOT/VERIFY/post-score states not implemented',
                      transport='dry_run', transmitted=False)
        return dict(coordinates=dict(target=target, avoid=avoid, selected=selected, status=status),
                    controller_input=dict(status='unavailable', imu_yaw_deg=None,
                                          altitude_m=None, sonar_held=None, battery_v=None,
                                          manual=None, heartbeat_age_ms=None),
                    mapping_status=self.mapping_status,
                    outgoing=dict(wire_v2_jsonl=encode_v2(message), legacy_jsonl=encode(message), draft_packet=packet,
                                  draft_jsonl=json.dumps(packet, separators=(',', ':'), allow_nan=False) + '\n'))


def mapping_identity_paths(args):
    paths = [Path(args.camera_mount), Path(args.target_sizes)]
    if args.lens_calibration:
        paths.append(Path(args.lens_calibration))
    return paths
