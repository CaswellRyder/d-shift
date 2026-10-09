import numpy as np
import pytest

from scripts.research_temporal_pan import (
    MS,
    SOURCE,
    episodes,
    render,
    score_view,
    simulate,
    summarize,
    temporal,
    view_truth,
    window,
)

COST = {"mser_direct": 320 * MS, "mser_bright": 245 * MS}
DELIVERY = 120 * MS


def blank(ticks):
    empty = dict(hits=[], missed=[], fp=0, ignored=0)
    return [{m: dict(empty) for m in COST} for _ in range(ticks)]


def detection(box, label="red_balloon", score=0.99, accepted=True):
    return dict(box=box, label=label, score=score, accepted=accepted)


def test_window_stays_inside_source_and_never_upsamples():
    for phase in (0.0, 0.9, 4.5, 9.9):
        for k in range(200):
            x0, y0, side = window(k / 10, phase)
            assert 320 <= side <= 416
            assert 0 <= x0 <= SOURCE - side and 0 <= y0 <= SOURCE - side


def test_full_window_render_matches_plain_resize():
    image = np.random.default_rng(0).integers(0, 255, (SOURCE, SOURCE, 3), np.uint8)
    assert render(image, (0, 0, SOURCE)).shape == (240, 320, 3)


def test_clipped_targets_become_ignore_regions():
    targets = [
        dict(label="red_balloon", source_box=[100, 100, 200, 200]),
        dict(label="blue_balloon", source_box=[300, 100, 400, 200]),
        dict(label="blue_balloon", source_box=[500, 500, 600, 600]),
    ]
    rows = view_truth(targets, (0, 0, 320))
    assert [(r["target"], r["present"]) for r in rows] == [(0, True), (1, False)]
    assert rows[0]["box"] == [100.0, 75.0, 200.0, 150.0]
    assert rows[1]["visible"] == pytest.approx(0.2)


def test_partial_target_detection_is_ignored_but_wrong_color_is_false_positive():
    truth = [
        dict(target=0, label="red_balloon", box=[0, 0, 40, 40], present=True),
        dict(target=1, label="blue_balloon", box=[200, 0, 220, 40], present=False),
    ]
    result = score_view(
        truth,
        [
            detection([1, 1, 40, 40]),
            detection([0, 0, 40, 40], score=0.9),  # duplicate of a matched target
            detection([200, 0, 230, 40], "blue_balloon"),
            detection([200, 0, 220, 40], "red_balloon"),
            detection([100, 100, 120, 120], accepted=False),
        ],
    )
    assert result == dict(hits=[0], missed=[], fp=2, ignored=1)


def test_simulation_takes_first_delivered_frame_after_each_result():
    periodic = simulate("periodic", blank(20), COST, DELIVERY)
    assert [(e["tick"], e["mode"]) for e in periodic] == [
        (0, "mser_direct"),
        (4, "mser_bright"),
        (7, "mser_direct"),
        (11, "mser_bright"),
        (14, "mser_direct"),
        (18, "mser_bright"),
    ]
    assert [e["reason"] for e in periodic[::2]] == ["startup_or_failure"] + ["elapsed_budget"] * 2
    full = simulate("full", blank(20), COST, DELIVERY)
    assert [e["tick"] for e in full] == [0, 4, 8, 12, 16]
    assert {e["mode"] for e in full} == {"mser_direct"}


def test_episode_latency_and_confirmed_age():
    assert episodes([False, True, True, False, True]) == [(1, 3), (4, 5)]
    present = {0: [False, False] + [True] * 6 + [False] * 12}
    scored = blank(20)
    scored[7]["mser_direct"] = dict(hits=[0], missed=[], fp=0, ignored=0)
    events = simulate("periodic", scored, COST, DELIVERY)
    (row,) = temporal(events, present)
    # Bright frame 4 misses; full frame 7 (result 1140 ms) is the first detection.
    assert row["processed"] == 2
    assert row["first_detection_ms"] == 940.0
    assert row["confirmed_ages_ms"] == [440.0]
    scored[4]["mser_bright"] = dict(hits=[0], missed=[], fp=0, ignored=0)
    (row,) = temporal(simulate("periodic", scored, COST, DELIVERY), present)
    assert row["first_detection_ms"] == 565.0
    assert row["confirmed_ages_ms"] == [365.0, 440.0]


def test_summary_separates_unprocessed_from_missed_episodes():
    present = {0: [True] + [False] * 19, 1: [False] * 4 + [True] * 2 + [False] * 14}
    scored = blank(20)
    scored[4]["mser_bright"] = dict(hits=[], missed=[1], fp=1, ignored=0)
    events = simulate("periodic", scored, COST, DELIVERY)
    run = dict(scene=0, ticks=20, events=events, episodes=temporal(events, present))
    summary = summarize([run], [["red_balloon", "blue_balloon"]])
    assert summary["episodes"] == 2 and summary["processed_episodes"] == 2
    assert summary["detected_episodes"] == 0
    assert summary["frame_counts"]["blue_balloon"]["fn"] == 1
    assert summary["false_positives"] == 1
    assert summary["first_detection"] is None
    assert [m["target"] for m in summary["missed_processed_episodes"]] == [0, 1]
