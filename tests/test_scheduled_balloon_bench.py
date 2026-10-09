import copy

import pytest

from dtr.research_search_schedule import SearchSchedule
from scripts import pi_balloon_schedule_bench as bench
from scripts.review_scheduled_balloon_search import validate_schedule


def observations():
    return dict(
        detections=[],
        processing_ms=1.0,
        search_ms=0.5,
        inference_ms=0.4,
        selection_ms=0.1,
        neural_calls=0,
    )


def fixture(monkeypatch):
    monkeypatch.setattr(bench, "process_frame", lambda *args: observations())
    clock = iter(
        [
            0,
            50_000_000,
            100_000_000,
            150_000_000,
            200_000_000,
            250_000_000,
            300_000_000,
            350_000_000,
        ]
    )
    scheduler = SearchSchedule()
    return [
        bench.scheduled_frame(None, None, None, scheduler, "periodic", lambda: next(clock))
        for _ in range(4)
    ]


def test_raw_timestamps_reproduce_dispatch_and_frame_cap(monkeypatch):
    rows = fixture(monkeypatch)
    result = validate_schedule("periodic", rows)
    assert result == dict(
        full_start_intervals_ms=[300.0],
        max_dispatch_lateness_ms=0.0,
        mode_counts=dict(mser_direct=2, mser_bright=2),
    )
    assert [r["reason"] for r in rows] == [
        "startup_or_failure",
        "bright_between_full",
        "bright_between_full",
        "frame_cap",
    ]


@pytest.mark.parametrize(
    "field,value",
    [
        ("mode", "mser_direct"),
        ("reason", "elapsed_budget"),
        ("due_lateness_ns", 1),
        ("start_ns", -1),
        ("end_ns", 0),
        ("scheduled_processing_ms", 1.0),
        ("processing_ms", float("nan")),
    ],
)
def test_tampered_scheduler_evidence_fails(monkeypatch, field, value):
    rows = copy.deepcopy(fixture(monkeypatch))
    rows[1][field] = value
    with pytest.raises(ValueError):
        validate_schedule("periodic", rows)


def test_processing_error_requires_full_search_on_next_attempt(monkeypatch):
    def fail(*args):
        raise RuntimeError("inference failed")

    scheduler = SearchSchedule()
    scheduler.begin(0)
    scheduler.finish(10)
    clock = iter([20, 30])
    monkeypatch.setattr(bench, "process_frame", fail)
    with pytest.raises(RuntimeError, match="inference failed"):
        bench.scheduled_frame(None, None, None, scheduler, "periodic", lambda: next(clock))
    assert scheduler.begin(40)["reason"] == "startup_or_failure"
    with pytest.raises(ValueError, match="Unknown"):
        bench.scheduled_frame(None, None, None, SearchSchedule(), "unknown")


def test_forced_full_control_uses_same_wrapper(monkeypatch):
    seen = []

    def record(rgb, predictor, mode, native):
        seen.append(mode)
        return observations()

    monkeypatch.setattr(bench, "process_frame", record)
    scheduler = SearchSchedule()
    for i in range(3):
        clock = iter([i * 10_000_000, i * 10_000_000 + 2_000_000])
        bench.scheduled_frame(None, None, None, scheduler, "full", lambda: next(clock))
    assert seen == ["mser_direct"] * 3
