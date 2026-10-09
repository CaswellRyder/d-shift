import pytest

from dtr.research_search_schedule import SearchSchedule


def frame(schedule, start, end, **kwargs):
    decision = schedule.begin(start, **kwargs)
    schedule.finish(end)
    return decision


def test_startup_and_frame_cap_even_at_high_rate():
    s = SearchSchedule()
    assert frame(s, 0, 10)["mode"] == "mser_direct"
    assert frame(s, 11, 20)["mode"] == "mser_bright"
    assert frame(s, 21, 30)["mode"] == "mser_bright"
    assert frame(s, 31, 40)["reason"] == "frame_cap"


def test_budget_includes_full_processing_time_and_reports_overrun():
    s = SearchSchedule()
    frame(s, 0, 700_000_000)
    d = frame(s, 710_000_000, 900_000_000)
    assert d["mode"] == "mser_direct" and d["reason"] == "elapsed_budget"
    assert d["due_lateness_ns"] == 210_000_000
    assert frame(s, 1_210_000_000, 1_220_000_000)["reason"] == "elapsed_budget"


def test_failed_search_forces_full_retry_and_never_advances_a_track():
    s = SearchSchedule()
    frame(s, 0, 10)
    assert s.begin(11)["mode"] == "mser_bright"
    s.finish(20, success=False)
    assert s.begin(21)["reason"] == "startup_or_failure"
    s.finish(30, success=False)
    assert frame(s, 31, 40)["mode"] == "mser_direct"
    assert not hasattr(s, "detections")


@pytest.mark.parametrize("bad", [-1, 1.0, True, None, float("nan")])
def test_invalid_clock_fails(bad):
    with pytest.raises(ValueError, match="nanoseconds"):
        SearchSchedule().begin(bad)


def test_ordering_and_pending_state_fail_closed():
    s = SearchSchedule()
    with pytest.raises(ValueError, match="pending"):
        s.finish(0)
    s.begin(10)
    with pytest.raises(ValueError, match="finish"):
        s.begin(11)
    with pytest.raises(ValueError, match="nanoseconds"):
        s.finish(9)
    s.finish(12)
    with pytest.raises(ValueError, match="nanoseconds"):
        s.begin(11)
    assert frame(s, 13, 14, force_full=True)["reason"] == "forced_full"


def test_forced_full_boolean_contract():
    with pytest.raises(ValueError, match="boolean"):
        SearchSchedule().begin(0, force_full=1)
