import json

import cv2
import numpy as np
import pytest

from dtr.flight import ColorTracker, Mission, MissionConfig, Track, encode

GREEN = (40, 200, 60)
PURPLE = (150, 60, 200)
ORANGE = (240, 110, 20)
YELLOW = (230, 220, 30)
FPS = 20


def frame(*shapes):
    rgb = np.full((240, 320, 3), 128, np.uint8)
    for kind, color, cx, cy, r in shapes:
        if kind == "disk":
            cv2.circle(rgb, (int(cx), int(cy)), r, color, -1)
        else:  # Hollow square rim, like a goal.
            cv2.rectangle(rgb, (int(cx - r), int(cy - r)), (int(cx + r), int(cy + r)), color, 3)
    return rgb


def box(cx, cy, r):
    return [cx - r, cy - r, cx + r, cy + r]


def seeded(label="green_balloon", cx=60, cy=120, r=10):
    tracker = ColorTracker()
    tracker.merge([dict(label=label, box=box(cx, cy, r), score=0.95)], frame_time=0.0, now=0.0)
    return tracker


def test_tracker_follows_moving_balloon_and_learns_velocity():
    tracker = seeded()
    for i in range(1, 21):
        tracker.update(frame(("disk", GREEN, 60 + 4 * i, 120, 10)), i / FPS)
        if i % 8 == 0:
            tracker.merge([dict(label="green_balloon", box=box(60 + 4 * i, 120, 10), score=0.9)], i / FPS, i / FPS)
    (track,) = tracker.tracks
    cx = (track.box[0] + track.box[2]) / 2
    assert abs(cx - 140) <= 2
    assert track.misses == 0
    assert track.velocity[0] == pytest.approx(80, rel=0.15)  # 4 px/frame at 20 fps.


def test_late_search_result_matches_the_moved_track_not_a_new_one():
    tracker = seeded()
    for i in range(1, 11):
        tracker.update(frame(("disk", GREEN, 60 + 6 * i, 120, 10)), i / FPS)
    # Full search finished now, but on the frame captured at 0.1 s (x = 72).
    tracker.merge([dict(label="green_balloon", box=box(72, 120, 10), score=0.9)], frame_time=0.1, now=0.5)
    (track,) = tracker.tracks
    assert track.confirmed == 0.1


def test_new_detection_is_refound_by_the_widened_window():
    tracker = ColorTracker()
    tracker.merge([dict(label="purple_balloon", box=box(100, 100, 8), score=0.9)], 0.0, 0.0)
    tracker.update(frame(("disk", PURPLE, 118, 104, 8)), 0.05)  # Moved 18 px since that frame.
    (track,) = tracker.tracks
    assert track.misses == 0 and abs((track.box[0] + track.box[2]) / 2 - 118) <= 2


def test_track_dropped_after_misses_and_after_confirm_timeout():
    tracker = seeded()
    empty = frame()
    for i in range(1, 6):
        tracker.update(empty, i / FPS)
    assert tracker.tracks == []

    tracker = seeded()
    for i in range(1, 40):  # Color keeps matching, but no search re-confirms it.
        tracker.update(frame(("disk", GREEN, 60, 120, 10)), i / FPS)
    assert tracker.tracks == []


def test_far_same_color_clutter_does_not_capture_track():
    tracker = seeded()
    for i in range(1, 6):
        tracker.update(frame(("disk", GREEN, 60, 120, 10), ("disk", GREEN, 280, 30, 12)), i / FPS)
    (track,) = tracker.tracks
    assert abs((track.box[0] + track.box[2]) / 2 - 60) <= 2


def test_track_does_not_latch_onto_bigger_same_color_blob():
    tracker = seeded(r=6)
    for i in range(1, 4):  # The balloon vanishes; a big green blob sits just beside it.
        tracker.update(frame(("disk", GREEN, 80, 120, 25)), i / FPS)
    assert all(t.misses > 0 for t in tracker.tracks)


def test_wrong_color_does_not_feed_a_track():
    tracker = seeded("purple_balloon")
    for i in range(1, 6):
        tracker.update(frame(("disk", GREEN, 60, 120, 10)), i / FPS)
    assert tracker.tracks == []


def test_box_at_returns_history_position():
    track = Track(1, "green_balloon", box(90, 90, 5), 0.0, 0.0)
    for i, x in enumerate((50, 60, 70, 80, 90)):
        track.history.append((i * 0.1, box(x, 90, 5)))
    assert track.box_at(0.25) == box(70, 90, 5)
    assert track.box_at(-1) == box(50, 90, 5)


def tracked(label, cx, cy, r, now, confirmed=None):
    t = Track(1, label, box(cx, cy, r), confirmed=now if confirmed is None else confirmed,
              seen=now, confirmed_wh=(2 * r, 2 * r))
    return [t]


def test_mission_search_track_lost_then_search():
    mission = Mission("orange")
    assert mission.step([], 0.0)["cmd"] == "SEARCH"
    message = mission.step(tracked("green_balloon", 200, 100, 10, 1.0), 1.0)
    assert message["cmd"] == "TRACK" and message["target"]["label"] == "green_balloon"
    assert message["target"]["x"] == pytest.approx(0.25) and message["target"]["size"] == pytest.approx(20 / 320, abs=1e-3)
    assert mission.step([], 1.5)["cmd"] == "LOST"
    assert mission.step([], 2.1)["cmd"] == "SEARCH"


def test_fan_on_needs_close_centered_and_fresh_balloon():
    config = MissionConfig()
    mission = Mission("orange", config=config)
    close = 0.35 * 320 / 2
    stale = mission.step(tracked("green_balloon", 160, 120, close, 5.0, confirmed=4.0), 5.0)
    assert stale["cmd"] == "TRACK" and not stale["fan"]
    off_center = mission.step(tracked("green_balloon", 300, 120, close, 5.1), 5.1)
    assert off_center["cmd"] == "TRACK"
    message = mission.step(tracked("purple_balloon", 160, 120, close, 5.2), 5.2)
    assert message["cmd"] == "FAN_ON" and message["fan"] and mission.phase == "capture"
    assert mission.step(tracked("purple_balloon", 160, 120, close, 5.25), 5.25)["cmd"] == "FAN_ON"
    lost = mission.step([], 5.3)  # No target: no capture command, but the fan stays on.
    assert lost["cmd"] == "LOST" and lost["fan"] and mission.phase == "capture"
    assert mission.step(tracked("purple_balloon", 300, 120, close, 5.35), 5.35)["cmd"] == "TRACK"
    mission.captured()
    assert mission.phase == "goal" and mission.step([], 5.4)["fan"]


def test_capture_timeout_returns_to_balloon_with_fan_off():
    mission = Mission("yellow", config=MissionConfig(capture_timeout_s=1.0))
    mission.step(tracked("green_balloon", 160, 120, 60, 0.0), 0.0)
    assert mission.phase == "capture"
    message = mission.step([], 1.5)
    assert mission.phase == "balloon" and not message["fan"] and message["cmd"] in ("SEARCH", "LOST")


def test_goal_phase_targets_own_color_avoids_opponent_and_enters_only_when_ready():
    mission = Mission("yellow")
    mission.captured()
    ours, theirs = "yellow_square", "orange_square"
    assert mission.step(tracked(theirs, 100, 100, 20, 1.0), 1.0)["cmd"] == "AVOID"
    both = tracked(theirs, 100, 100, 40, 1.1) + tracked(ours, 220, 100, 15, 1.1)
    message = mission.step(both, 1.1)
    assert message["cmd"] == "TRACK" and message["target"]["label"] == ours
    big = 0.5 * 320 / 2
    assert mission.step(tracked(ours, 160, 120, big, 2.0, confirmed=1.0), 2.0)["cmd"] == "TRACK"
    assert mission.step(tracked(ours, 160, 120, big, 2.1), 2.1)["cmd"] == "ENTER"
    mission.scored()
    assert mission.phase == "balloon" and not mission.fan


def test_mission_rejects_unknown_goal_color_and_encodes_one_line():
    with pytest.raises(ValueError):
        Mission("purple")
    mission = Mission("orange")
    mission.set_goal_color("yellow")
    line = encode(mission.step(tracked("green_balloon", 160, 120, 10, 1.0), 1.0))
    assert line.endswith("\n") and line.count("\n") == 1
    decoded = json.loads(line)
    assert decoded["v"] == 1 and decoded["goal_color"] == "yellow" and decoded["seq"] == 1
    assert set(decoded["target"]) == {"label", "x", "y", "size", "age_ms", "seen_ms"}


def test_goal_rim_is_tracked_by_color():
    tracker = ColorTracker()
    tracker.merge([dict(label="orange_square", box=box(160, 80, 20), score=0.9)], 0.0, 0.0)
    for i in range(1, 6):
        tracker.update(frame(("rim", ORANGE, 160 + 2 * i, 80, 20), ("rim", YELLOW, 60, 160, 20)), i / FPS)
    (track,) = tracker.tracks
    assert track.label == "orange_square" and abs((track.box[0] + track.box[2]) / 2 - 170) <= 2


def test_repeated_small_detections_snap_swollen_track_and_prevent_fan_on():
    tracker = seeded(cx=160, r=20)
    (track,) = tracker.tracks
    detection_box = box(160, 120, 20)
    for i in range(1, 9):
        now = i * 0.5
        # Color tracking has drifted onto a 121 px blob; full search is 0.25 s late.
        track.box, track.misses, track.seen = box(160, 120, 60.5), 0, now
        track.history.append((now - 0.25, list(track.box)))
        tracker.merge([dict(label=track.label, box=detection_box, score=0.9)], now - 0.25, now)
        assert tracker.tracks == [track]
        assert track.box == detection_box
        assert track.confirmed_wh == (40, 40)
        assert track.misses == 1
        assert Mission("orange").step(tracker.tracks, now)["cmd"] != "FAN_ON"
        # A plausible color hit can recover the snapped track, but remains too small.
        tracker.update(frame(("disk", GREEN, 160, 120, 20)), now + 0.05)
        assert track.misses == 0
        assert Mission("orange").step(tracker.tracks, now + 0.05)["cmd"] == "TRACK"


def test_consistent_late_detection_refreshes_confirmation_and_keeps_current_box():
    tracker = seeded(cx=100, r=20)
    (track,) = tracker.tracks
    track.history.append((0.5, box(110, 120, 20)))
    track.box, track.seen, track.misses = box(150, 120, 20), 1.0, 0
    track.velocity = (80.0, 0.0)
    tracker.merge([dict(label=track.label, box=box(110, 120, 20), score=0.88)], 0.5, 1.0)
    assert tracker.tracks == [track]
    assert track.confirmed == 0.5 and track.score == 0.88
    assert track.box == box(150, 120, 20) and track.velocity == (80.0, 0.0)
    assert track.seen == 1.0 and track.misses == 0
    assert track.confirmed_wh == (40, 40)
    # An older consistent result must not move confirmation time backwards.
    tracker.merge([dict(label=track.label, box=box(110, 120, 20), score=0.9)], 0.4, 1.1)
    assert track.confirmed == 0.5


def test_inconsistent_overlap_snaps_without_motion_correction():
    tracker = seeded(cx=160, r=60)
    (track,) = tracker.tracks
    track.confirmed, track.velocity, track.misses = 0.3, (100.0, -30.0), 0
    detection_box = box(170, 120, 20)
    tracker.merge([dict(label=track.label, box=detection_box, score=0.87)], 0.5, 1.0)
    assert tracker.tracks == [track]
    assert track.box == detection_box and track.velocity == (0.0, 0.0)
    assert track.misses == 1 and track.confirmed == 0.5
    assert track.confirmed_wh == (40, 40) and track.score == 0.87
    assert track.history[-1] == (1.0, detection_box)


@pytest.mark.parametrize("label,current_side,confirmed_side,command", [
    ("green_balloon", 120, 85, "FAN_ON"),
    ("orange_square", 160, 110, "ENTER"),
])
def test_mission_size_gate_caps_growth_since_confirmation(
        label, current_side, confirmed_side, command):
    mission = Mission("orange")
    if label.endswith("square"):
        mission.captured()
    (track,) = tracked(label, 160, 120, current_side / 2, 1.0)
    track.confirmed_wh = (40.0, 40.0)
    message = mission.step([track], 1.0)
    assert message["cmd"] == "TRACK"
    assert message["target"]["size"] == pytest.approx(current_side / 320)
    track.confirmed_wh = (confirmed_side, confirmed_side)
    assert mission.step([track], 1.1)["cmd"] == command


def test_new_track_records_classifier_confirmed_dimensions():
    (track,) = seeded(r=12).tracks
    assert track.confirmed_wh == (24, 24)


def test_equal_iou_candidates_choose_nearest_center():
    tracker = ColorTracker()
    far = Track(1, "green_balloon", box(110, 120, 5), 0.0, 0.0)
    near = Track(2, "green_balloon", box(125, 120, 5), 0.0, 0.0)
    tracker.tracks = [far, near]
    tracker.next_id = 3
    # Both have zero IoU and are within one detection side; nearest must win.
    tracker.merge([dict(label=far.label, box=box(160, 120, 30), score=0.9)], 0.5, 1.0)
    assert far.confirmed == 0.0 and near.confirmed == 0.5
    assert near.box == box(160, 120, 30) and near.misses == 1


def test_tiny_overlap_and_distant_center_do_not_match():
    tracker = seeded(cx=80, r=60)
    (track,) = tracker.tracks
    tracker.merge([dict(label=track.label, box=box(145, 120, 10), score=0.9)], 0.5, 1.0)
    assert len(tracker.tracks) == 2
    assert track.confirmed == 0.0 and track.box == box(80, 120, 60)


@pytest.mark.parametrize("offset,consistent", [(4, True), (5, False)])
def test_small_target_center_fallback_still_requires_close_center(offset, consistent):
    tracker = seeded(cx=100, r=4)
    (track,) = tracker.tracks
    track.box = [96, 117, 104, 123]  # Small 8x6 target: shifted IoU is below 0.3.
    track.history.append((0.5, list(track.box)))
    track.misses = 0
    detection_box = box(100 + offset, 120, 4)
    tracker.merge([dict(label=track.label, box=detection_box, score=0.9)], 0.5, 1.0)
    assert tracker.tracks == [track]
    assert track.confirmed_wh == (8, 8)
    assert track.misses == (0 if consistent else 1)
    assert track.box == ([96, 117, 104, 123] if consistent else detection_box)


@pytest.mark.parametrize("min_iou,max_ratio", [(0.9, 1.5), (0.3, 1.1)])
def test_match_consistency_thresholds_are_configurable(min_iou, max_ratio):
    tracker = ColorTracker(min_match_iou=min_iou, max_size_ratio=max_ratio)
    tracker.merge([dict(label="green_balloon", box=box(160, 120, 24), score=0.9)], 0.0, 0.0)
    (track,) = tracker.tracks
    track.misses = 0
    tracker.merge([dict(label=track.label, box=box(160, 120, 20), score=0.85)], 0.5, 1.0)
    assert track.box == box(160, 120, 20) and track.misses == 1


@pytest.mark.parametrize("label,current_side,confirmed_side", [
    ("green_balloon", 171, 70),
    ("orange_square", 195, 100),
])
def test_swollen_track_never_triggers_actuation(label, current_side, confirmed_side):
    mission = Mission("orange")
    if label.endswith("square"):
        mission.captured()
    (track,) = tracked(label, 160, 120, current_side / 2, 1.0)
    track.confirmed_wh = (confirmed_side, confirmed_side)
    assert mission.step([track], 1.0)["cmd"] == "TRACK"


def test_capture_never_sends_fan_on_without_ready_target():
    mission = Mission("orange")
    mission.step(tracked("green_balloon", 160, 120, 60, 0.25), 0.25)
    assert mission.phase == "capture"
    assert mission.step([], 3.0)["cmd"] != "FAN_ON"
    stale = mission.step(tracked("green_balloon", 160, 120, 60, 3.5, confirmed=2.0), 3.5)
    assert stale["cmd"] == "TRACK" and stale["fan"]


def test_duplicate_removal_keeps_newest_classifier_correction():
    tracker = ColorTracker()
    first = Track(1, "green_balloon", box(160, 120, 60.5), confirmed=0.1, seen=0.5, confirmed_wh=(70, 70))
    second = Track(2, "green_balloon", box(160, 120, 25), confirmed=0.1, seen=0.5, confirmed_wh=(70, 70))
    first.history.append((0.25, box(160, 120, 40)))  # 80 px then: the 40 px result snaps it.
    second.history.append((0.25, box(200, 120, 50)))
    tracker.tracks, tracker.next_id = [first, second], 3
    tracker.merge([dict(label="green_balloon", box=box(160, 120, 20), score=0.9)], 0.25, 0.5)
    (track,) = tracker.tracks  # The snapped track (misses=1) overlaps the other at IoU 0.64.
    assert track is first and track.confirmed == 0.25 and track.confirmed_wh == (40, 40)


def test_snapped_track_reacquires_a_grown_target():
    tracker = seeded(cx=160, r=45)
    (track,) = tracker.tracks
    track.misses = 0
    tracker.merge([dict(label=track.label, box=box(160, 120, 20), score=0.9)], 0.5, 1.0)
    assert track.box == box(160, 120, 20) and track.misses == 1
    tracker.update(frame(("disk", GREEN, 160, 120, 40)), 1.05)  # 81 px, twice the snap.
    assert track.misses == 0 and 78 <= track.box[2] - track.box[0] <= 84


def test_older_result_does_not_overwrite_newer_confirmation():
    tracker = seeded(label="orange_square", cx=160, r=20)
    (track,) = tracker.tracks
    track.history.append((0.4, box(160, 120, 50)))
    track.history.append((0.8, box(160, 120, 20)))
    tracker.merge([dict(label=track.label, box=box(160, 120, 20), score=0.9)], 0.8, 1.0)
    tracker.merge([dict(label=track.label, box=box(160, 120, 50), score=0.8)], 0.4, 1.2)
    assert track.confirmed == 0.8 and track.confirmed_wh == (40, 40) and track.score == 0.9


@pytest.mark.parametrize("label,confirmed_wh,current_wh", [
    ("orange_square", (90, 150), (160, 150)),
    ("green_balloon", (70, 100), (110, 100)),
])
def test_width_swelling_alone_blocks_actuation(label, confirmed_wh, current_wh):
    mission = Mission("orange")
    if label.endswith("square"):
        mission.captured()
    (track,) = tracked(label, 160, 120, 10, 1.0)
    w, h = current_wh
    track.box = [160 - w / 2, 120 - h / 2, 160 + w / 2, 120 + h / 2]
    track.confirmed_wh = confirmed_wh
    assert mission.step([track], 1.0)["cmd"] == "TRACK"


def test_late_result_narrower_than_swollen_history_snaps_the_track():
    tracker = seeded(label="orange_square", cx=160, r=45)
    (track,) = tracker.tracks
    track.misses = 0
    track.history.append((0.5, [80.0, 45.0, 240.0, 195.0]))  # 160x150 color box then.
    tracker.merge([dict(label=track.label, box=[115, 45, 205, 195], score=0.9)], 0.5, 1.0)
    assert track.box == [115, 45, 205, 195] and track.misses == 1
    assert track.confirmed_wh == (90, 150)
