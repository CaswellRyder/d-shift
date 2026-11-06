"""Synthetic telemetry checks: no camera, hardware transport, or controller claims."""
from dtr.target_geometry import load_target_sizes

from copy import deepcopy
import json
import zipfile

import cv2
import numpy as np
import pytest

from dtr.flight import ColorTracker, Mission, Track, encode
from dtr.flight_lab import CaptureBuffer, export_capture, validate_capture
from dtr.flight_pipeline import FlightPipeline
from dtr.flight_telemetry import FlightTelemetry
from dtr.lens import LensModel
from dtr.target_geometry import CameraMount, TargetGeometry


def track(label="green_balloon", box=(180., 70., 220., 110.), now=10., **changes):
    result = dict(id=1, label=label, box=list(box), score=.95,
                  confirmed=now, seen=now, misses=0)
    result.update(changes)
    return result


def row(cmd="TRACK", tracks=None, label="green_balloon", phase="balloon", now=10.):
    tracks = deepcopy([track(label=label, now=now)] if tracks is None else tracks)
    actual_tracks = []
    for ident, snapshot in enumerate(tracks, 1):
        snapshot["id"] = ident
        x1, y1, x2, y2 = snapshot["box"]
        actual_tracks.append(Track(ident, snapshot["label"], snapshot["box"],
                                   confirmed=snapshot["confirmed"], seen=snapshot["seen"],
                                   score=snapshot["score"], misses=snapshot["misses"],
                                   confirmed_wh=(x2 - x1, y2 - y1)))
    mission = Mission("orange")
    if phase == "goal":
        mission.captured()
    message = mission.step(actual_tracks, now)
    message["seq"] = 42
    message["cmd"] = cmd
    if cmd == "FAN_ON":
        message["fan"] = True
    return dict(phase_before=phase, message=message, tracks=tracks,
                search_result=None, submitted=True, selected_track_id=mission.selected_track_id)


@pytest.fixture
def telemetry():
    return FlightTelemetry()


def test_camera_body_axes_and_pointing_signs_are_consistent(telemetry):
    observed = telemetry.observe(row(), 10.)
    target = observed["coordinates"]["target"]
    cam_right, cam_down, cam_forward = target["cam_xyz_m"]
    forward, right, up = target["body_xyz_m"]
    assert cam_right > 0 and cam_down < 0 and cam_forward > 0
    assert (forward, right, up) == pytest.approx((cam_forward, cam_right, -cam_down))
    assert target["bearing_deg"] > 0 and target["elevation_deg"] > 0
    packet = observed["outgoing"]["draft_packet"]
    assert packet["relative_guidance"]["yaw_error_deg"] > 0
    assert packet["relative_guidance"]["climb_error_deg"] > 0


def test_centre_target_has_zero_lateral_pointing_error(telemetry):
    observed = telemetry.observe(row(tracks=[track(box=(140., 100., 180., 140.))]), 10.)
    target = observed["coordinates"]["target"]
    assert target["cam_xyz_m"][:2] == pytest.approx((0., 0.), abs=1e-12)
    assert target["yaw_err_deg"] == pytest.approx(0., abs=1e-12)
    assert target["climb_err_deg"] == pytest.approx(0., abs=1e-12)


def test_centred_target_three_metres_ahead_has_current_xyz_not_frame_delta(telemetry):
    radius = load_target_sizes()['green_balloon']['width_m'] / 2  # Narrower axis governs.
    half_pixels = telemetry.lens.K[0, 0] * np.arcsin(radius / 3.)
    sample = row(tracks=[track(box=(160-half_pixels, 120-half_pixels,
                                   160+half_pixels, 120+half_pixels))])
    packet = telemetry.observe(sample, 10.)['outgoing']['draft_packet']
    assert packet['delta_axes'] == 'x_right,y_up,z_forward'
    assert packet['camera_delta_m'] == pytest.approx(dict(x_right_m=0., y_up_m=0., z_forward_m=3.))
    assert packet['navigation_delta_m'] == pytest.approx(packet['camera_delta_m'])
    assert packet['camera_error_normalized'] == dict(x_right=0., y_up=0.)
    assert packet['desired_standoff_m'] is None and packet['forward_error_m'] is None


def test_packet_right_up_forward_reorders_internal_body_axes(telemetry):
    packet = telemetry.observe(row(), 10.)['outgoing']['draft_packet']
    forward, right, up = packet['target']['body_xyz_m']
    assert packet['navigation_delta_m'] == dict(x_right_m=right, y_up_m=up, z_forward_m=forward)
    assert packet['navigation_delta_m']['x_right_m'] > 0
    assert packet['navigation_delta_m']['y_up_m'] > 0


def test_legacy_wire_line_stays_byte_exact_and_draft_roundtrips(telemetry):
    sample = row()
    original = deepcopy(sample)
    observed = telemetry.observe(sample, 10.)
    assert sample == original
    outgoing = observed["outgoing"]
    assert outgoing["legacy_jsonl"] == encode(sample["message"])
    assert json.loads(outgoing["legacy_jsonl"]) == sample["message"]
    assert outgoing["draft_jsonl"].endswith("\n")
    assert outgoing["draft_jsonl"].count("\n") == 1
    assert json.loads(outgoing["draft_jsonl"]) == json.loads(json.dumps(outgoing["draft_packet"]))
    json.dumps(observed, allow_nan=False)


def test_unavailable_controller_inputs_are_null_and_never_motor_commands(telemetry):
    observed = telemetry.observe(row(cmd="FAN_ON"), 10.)
    controller = observed["controller_input"]
    assert controller["status"] == "unavailable"
    for name in ("imu_yaw_deg", "altitude_m", "sonar_held", "battery_v", "manual", "heartbeat_age_ms"):
        assert controller[name] is None
    packet = observed["outgoing"]["draft_packet"]
    assert packet["schema"] == "pi-esp32-guidance-draft-v1"
    assert packet["seq"] == 42 and packet["frame_time_s"] == 10.
    assert packet["timestamp_kind"] == "post_acquisition_monotonic"
    assert packet["relative_guidance"]["forward_thrust"] is None
    assert packet["world_heading_deg"] is None and packet["actuator_command"] is None
    assert packet["control_ready"] is False
    assert packet["transport"] == "dry_run" and packet["transmitted"] is False
    assert "motor_pwm" not in packet and "servo_angle_deg" not in packet
    assert "imu_and_controller_telemetry_unavailable" in packet["blockers"]


def test_default_uncalibrated_coordinates_stay_estimates(telemetry):
    observed = telemetry.observe(row(), 10.)
    target = observed["coordinates"]["target"]
    assert not target["lens_calibrated"]
    assert not target["mount_calibrated"]
    assert target["size_calibrated"]  # Balloon measured 16 x 18 in (2026-11-06).
    assert not target["range_qualified"]
    blockers = observed["outgoing"]["draft_packet"]["blockers"]
    assert {"lens_uncalibrated", "camera_mount_unmeasured"} <= set(blockers)
    assert "target_size_unmeasured" not in blockers


def test_calibration_flags_cannot_approve_an_unapproved_controller():
    lens = LensModel.ideal_equidistant()
    lens.calibrated = True
    geometry = TargetGeometry(lens, mount=CameraMount(calibrated=True))
    geometry.sizes["green_balloon"]["calibrated"] = True
    observed = FlightTelemetry(geometry).observe(row(), 10.)
    target = observed["coordinates"]["target"]
    assert target["lens_calibrated"] and target["mount_calibrated"] and target["size_calibrated"]
    packet = observed["outgoing"]["draft_packet"]
    assert packet["control_ready"] is False
    assert "esp32_control_contract_not_approved" in packet["blockers"]
    assert "lens_uncalibrated" not in packet["blockers"]
    assert "camera_mount_unmeasured" not in packet["blockers"]


@pytest.mark.parametrize("label", ["orange_circle", "orange_square", "orange_triangle",
                                   "yellow_circle", "yellow_square", "yellow_triangle"])
def test_goal_pose_uncertainty_is_not_hidden_by_known_goal_size(telemetry, label):
    observed = telemetry.observe(row(label=label, phase="goal"), 10.)
    target = observed["coordinates"]["target"]
    assert target["size_calibrated"]
    assert target["goal_pose_ambiguous"] and not target["range_qualified"]
    assert "goal_pose_range_ambiguous" in observed["outgoing"]["draft_packet"]["blockers"]


def test_avoidance_is_not_a_navigation_target_or_relative_steering_command(telemetry):
    observed = telemetry.observe(row(cmd="AVOID", label="yellow_square", phase="goal"), 10.)
    coordinates = observed["coordinates"]
    assert coordinates["target"] is None
    assert coordinates["avoid"]["label"] == "yellow_square"
    assert coordinates["selected"] == coordinates["avoid"]
    packet = observed["outgoing"]["draft_packet"]
    assert packet["target"] is None and packet["avoid"] is not None
    assert packet["relative_guidance"] == dict(yaw_error_deg=None, climb_error_deg=None,
                                               forward_thrust=None)


@pytest.mark.parametrize("command", ["LOST", "SEARCH"])
def test_no_published_target_means_no_coordinates_or_steering(telemetry, command):
    sample = row(cmd=command, tracks=[track(misses=1)])
    sample["message"]["target"] = None
    observed = telemetry.observe(sample, 10.)
    assert observed["coordinates"]["selected"] is None
    packet = observed["outgoing"]["draft_packet"]
    assert packet["target"] is None and packet["avoid"] is None
    assert all(value is None for value in packet["relative_guidance"].values())


def test_other_label_track_cannot_supply_geometry_for_the_published_target(telemetry):
    sample = row(tracks=[track(label="purple_balloon")])
    sample["message"]["target"] = dict(label="green_balloon", x=0., y=0., size=.1,
                                       age_ms=0, seen_ms=0)
    observed = telemetry.observe(sample, 10.)
    assert observed["coordinates"]["selected"] is None
    assert observed["outgoing"]["draft_packet"]["target"] is None


@pytest.mark.parametrize("age,seen,allowed", [(.8, .1, True), (.801, 0., False), (0., .101, False)])
def test_pointing_requires_fresh_search_confirmation_and_current_color_hit(
        telemetry, age, seen, allowed):
    sample = row(tracks=[track(now=20., confirmed=20. - age, seen=20. - seen)], now=20.)
    observed = telemetry.observe(sample, 20.)
    packet = observed["outgoing"]["draft_packet"]
    assert (packet["relative_guidance"]["yaw_error_deg"] is not None) is allowed
    if not allowed:
        assert "target_stale" in packet["blockers"]
        assert all(value is None for value in packet["relative_guidance"].values())


def test_clipped_target_is_flagged_and_cannot_produce_pointing_commands(telemetry):
    observed = telemetry.observe(row(tracks=[track(box=(0., 100., 40., 140.))]), 10.)
    target = observed["coordinates"]["target"]
    assert target["clipped"] and target["range_is_upper_bound"]
    packet = observed["outgoing"]["draft_packet"]
    assert "target_clipped" in packet["blockers"]
    assert all(value is None for value in packet["relative_guidance"].values())


def test_geometry_uses_the_same_width_then_confirmation_priority_as_mission(telemetry):
    candidates = [track(box=(145., 105., 175., 135.), confirmed=9.99),
                  track(box=(170., 60., 230., 120.), confirmed=9.8),
                  track(label="purple_balloon", box=(20., 20., 140., 140.))]
    observed = telemetry.observe(row(tracks=candidates), 10.)
    assert observed["coordinates"]["selected"]["box_px"] == tuple(candidates[2]["box"])


def test_equal_width_geometry_uses_the_newer_full_search_confirmation(telemetry):
    candidates = [track(box=(140., 100., 180., 140.), confirmed=9.7),
                  track(box=(180., 70., 220., 110.), confirmed=9.95)]
    observed = telemetry.observe(row(tracks=candidates), 10.)
    assert observed["coordinates"]["selected"]["box_px"] == tuple(candidates[1]["box"])


@pytest.mark.parametrize("with_selected_id", [True, False])
def test_equal_range_avoidance_goals_use_newest_confirmation(
        telemetry, with_selected_id):
    tracker = ColorTracker()
    tracker.tracks = [
        Track(11, "yellow_square", [30., 100., 70., 140.], confirmed=9.5, seen=10.,
              score=.95, confirmed_wh=(40., 40.)),
        Track(12, "yellow_square", [230., 100., 270., 140.], confirmed=9.9, seen=10.,
              score=.99, confirmed_wh=(40., 40.)),
    ]
    image = np.zeros((240, 320, 3), np.uint8)
    for left in (30, 230):
        cv2.rectangle(image, (left, 100), (left + 40, 140), (230, 220, 30), 3)
    mission = Mission("orange")
    mission.captured()
    sample = FlightPipeline(StubWorker(), tracker, mission).step(image, 10.)
    assert sample["message"]["cmd"] == "AVOID"
    assert sample["message"]["target"]["x"] > 0
    assert sample["selected_track_id"] == 12
    expected = next(item for item in sample["tracks"] if item["id"] == 12)
    if not with_selected_id:
        sample.pop("selected_track_id")
    observed = telemetry.observe(sample, 10.)
    assert observed["coordinates"]["target"] is None
    assert observed["coordinates"]["avoid"]["box_px"] == tuple(expected["box"])
    assert observed["coordinates"]["avoid"]["body_xyz_m"][1] > 0
    assert observed["outgoing"]["legacy_jsonl"] == encode(sample["message"])
    assert all(value is None for value in observed["outgoing"]["draft_packet"]["relative_guidance"].values())


def test_missing_selected_identity_cannot_substitute_a_different_track(telemetry):
    sample = row()
    sample["selected_track_id"] = 999
    observed = telemetry.observe(sample, 10.)
    assert observed["coordinates"]["selected"] is None
    assert "no_valid_target_geometry" in observed["outgoing"]["draft_packet"]["blockers"]


def test_old_row_fallback_requires_the_actual_published_normalized_point(telemetry):
    sample = row()
    sample.pop("selected_track_id")
    sample["message"]["target"]["x"] = -.987
    observed = telemetry.observe(sample, 10.)
    assert observed["coordinates"]["selected"] is None


def test_missed_track_does_not_produce_geometry_for_a_published_label(telemetry):
    sample = row(tracks=[track(misses=1)])
    observed = telemetry.observe(sample, 10.)
    assert observed["coordinates"]["selected"] is None
    assert "no_valid_target_geometry" in observed["outgoing"]["draft_packet"]["blockers"]


@pytest.mark.parametrize("case", ["unsupported_label", "invalid_box", "untrusted_corner"])
def test_bad_geometry_becomes_unavailable_without_crashing_the_frame(telemetry, case):
    sample = row()
    if case == "unsupported_label":
        sample["message"]["target"]["label"] = "unknown_shape"
        sample["tracks"][0]["label"] = "unknown_shape"
    elif case == "invalid_box":
        sample["tracks"][0]["box"] = [100., 100., 90., 120.]
    else:
        sample["tracks"][0]["box"] = [0., 0., 10., 10.]
    observed = telemetry.observe(sample, 10.)
    assert observed["coordinates"]["selected"] is None
    assert observed["coordinates"]["status"].startswith("geometry_unavailable")
    assert all(value is None for value in observed["outgoing"]["draft_packet"]["relative_guidance"].values())
    json.dumps(observed, allow_nan=False)


def test_lens_inversion_failure_is_a_reported_unavailable_coordinate(monkeypatch):
    geometry = TargetGeometry(LensModel.ideal_equidistant())
    telemetry = FlightTelemetry(geometry)

    def fail(*args, **kwargs):
        raise ValueError("Synthetic lens inverse failure")

    monkeypatch.setattr(geometry, "target", fail)
    observed = telemetry.observe(row(), 10.)
    assert observed["coordinates"]["selected"] is None
    assert "Synthetic lens inverse failure" in observed["coordinates"]["status"]


def test_tape_measurements_are_not_inputs_to_estimated_camera_coordinates(telemetry):
    first, second = row(), row()
    first["distance_m"], second["distance_m"] = 1., 99.
    assert telemetry.observe(first, 10.) == telemetry.observe(second, 10.)


def test_mapping_has_finite_json_angles_correct_axes_and_untrusted_corners(telemetry):
    mapping = telemetry.mapping()
    json.dumps(mapping, allow_nan=False)
    assert (mapping["width"], mapping["height"]) == (320, 240)
    samples = {(item["u"], item["v"]): item for item in mapping["samples"]}
    assert samples[160, 120]["bearing_deg"] == pytest.approx(0.)
    assert samples[160, 120]["elevation_deg"] == pytest.approx(0.)
    assert samples[320, 120]["bearing_deg"] > 0
    assert samples[160, 0]["elevation_deg"] > 0
    for corner in ((0, 0), (320, 0), (0, 240), (320, 240)):
        assert samples[corner]["trusted"] is False
    assert "right, down, forward" in mapping["axes"]["camera"]
    assert "forward, right, up" in mapping["axes"]["body"]
    assert mapping["lens_calibrated"] is False


def test_mapping_is_precomputed_and_not_reinverted_for_each_request(telemetry, monkeypatch):
    expected = deepcopy(telemetry.mapping())

    def fail(*args, **kwargs):
        raise AssertionError("Mapping unexpectedly recomputed lens rays")

    monkeypatch.setattr(telemetry.lens, "rays", fail)
    assert telemetry.mapping() == expected


def test_mapping_marks_failed_inversions_untrusted_and_keeps_valid_centre():
    lens = LensModel.ideal_equidistant()
    lens.D[0, 0] = -.5
    mapping = FlightTelemetry(TargetGeometry(lens)).mapping()
    json.dumps(mapping, allow_nan=False)
    failed = [sample for sample in mapping["samples"] if sample["bearing_deg"] is None]
    assert failed
    assert all(sample["trusted"] is False and sample["elevation_deg"] is None for sample in failed)
    centre = next(sample for sample in mapping["samples"] if (sample["u"], sample["v"]) == (160, 120))
    assert centre["trusted"] and centre["bearing_deg"] == pytest.approx(0.)


def test_body_pointing_includes_camera_mount_rotation_without_inventing_world_yaw():
    geometry = TargetGeometry(LensModel.ideal_equidistant(),
                              mount=CameraMount(yaw_deg=15., calibrated=True))
    sample = row(tracks=[track(box=(140., 100., 180., 140.))])
    observed = FlightTelemetry(geometry).observe(sample, 10.)
    target = observed["coordinates"]["target"]
    assert target["bearing_deg"] == pytest.approx(0.)
    assert target["yaw_err_deg"] == pytest.approx(15.)
    packet = observed["outgoing"]["draft_packet"]
    assert packet["relative_guidance"]["yaw_error_deg"] == pytest.approx(15.)
    assert packet["world_heading_deg"] is None


class StubWorker:
    def poll(self):
        return None

    def submit(self, *args):
        return True


def seeded_tracker():
    tracker = ColorTracker()
    tracker.tracks = [Track(1, "green_balloon", [180., 70., 220., 110.],
                            confirmed=10., seen=10., score=.95, confirmed_wh=(40., 40.))]
    return tracker


def test_optional_pipeline_telemetry_preserves_the_legacy_message_and_submission(telemetry):
    plain = FlightPipeline(StubWorker(), seeded_tracker(), Mission("orange"))
    augmented = FlightPipeline(StubWorker(), seeded_tracker(), Mission("orange"), telemetry=telemetry)
    rgb = np.zeros((240, 320, 3), np.uint8)
    rgb[70:110, 180:220] = (40, 200, 60)
    expected = plain.step(rgb, 10.05)
    actual = augmented.step(rgb, 10.05)
    assert actual["message"] == expected["message"]
    assert actual["tracks"] == expected["tracks"]
    assert actual["submitted"] == expected["submitted"]
    assert actual["flight_telemetry"]["outgoing"]["legacy_jsonl"] == encode(expected["message"])


def test_export_saves_the_live_wire_strings_without_reobserving(tmp_path, telemetry, monkeypatch):
    metadata = validate_capture(dict(distance_m=2.345, target_label="green_balloon",
                                    expected_count=1, scene="telemetry", notes=""), "balloon", "orange")
    capture = CaptureBuffer()
    capture.start(metadata, 10.)
    legacy_lines, draft_lines = [], []
    rgb = np.zeros((240, 320, 3), np.uint8)
    for seq, now in enumerate((10., 10.05, 10.1), 1):
        sample = row(now=now)
        sample["message"]["seq"] = seq
        sample["flight_telemetry"] = telemetry.observe(sample, now)
        outgoing = sample["flight_telemetry"]["outgoing"]
        # Deliberate JSON whitespace is evidence: export must preserve recorded bytes.
        outgoing["draft_jsonl"] = json.dumps(outgoing["draft_packet"], sort_keys=True) + "\n"
        legacy_lines.append(outgoing["legacy_jsonl"])
        draft_lines.append(outgoing["draft_jsonl"])
        assert capture.append(rgb, sample, now)
    assert not capture.append(rgb, row(now=15.), 15.)

    def forbid_reobservation(*args, **kwargs):
        raise AssertionError("Export reran geometry/controller telemetry")

    monkeypatch.setattr(FlightTelemetry, "observe", forbid_reobservation)
    summary = export_capture(capture, tmp_path, {"source": "replay"})
    folder = tmp_path / summary["id"]
    assert (folder / "wire-v1.jsonl").read_text() == "".join(legacy_lines)
    assert (folder / "esp32-draft.jsonl").read_text() == "".join(draft_lines)
    with zipfile.ZipFile(folder / "download.zip") as archive:
        assert archive.read("wire-v1.jsonl").decode() == "".join(legacy_lines)
        assert archive.read("esp32-draft.jsonl").decode() == "".join(draft_lines)
