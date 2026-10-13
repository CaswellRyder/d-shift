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
    t = Track(1, label, box(cx, cy, r), confirmed=now if confirmed is None else confirmed, seen=now)
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
    assert mission.step([], 5.3)["cmd"] == "FAN_ON"  # Keeps asking while capturing.
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
