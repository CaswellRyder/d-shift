import pytest

from scripts.evaluate_pixel_recovery import dispatch, matched_targets, target_metrics


def test_full_search_recovers_a_systematic_pixel_miss():
    costs = dict(mser=280_000_000, pixel2=70_000_000)
    matches = dict(mser={0}, pixel2=set())
    pixel = target_metrics(dispatch("pixel", costs), matches, 0, 3_000_000_000)
    recovered = target_metrics(dispatch("recovery", costs), matches, 0, 3_000_000_000)
    assert pixel["confirmations"] == 0
    assert recovered["confirmations"] >= 4
    assert recovered["max_confirmation_gap_ms"] <= 600


def test_results_after_window_do_not_count():
    assert dispatch("full", dict(mser=4_000_000_000, pixel2=10)) == []


def test_timestamps_and_between_search_cap():
    events = dispatch("recovery", dict(mser=100_000_000, pixel2=10_000_000))
    between = 0
    for i, event in enumerate(events):
        assert event["sensor_ns"] <= event["start_ns"] < event["result_ns"]
        if i:
            assert event["start_ns"] >= events[i-1]["result_ns"]
        between = 0 if event["method"] == "mser" else between + 1
        assert between <= 2


def test_missing_observation_has_no_invented_age():
    result = target_metrics([], dict(mser=set(), pixel2=set()), 0, 3_000_000_000)
    assert result["first_detection_ms"] is None
    assert result["end_confirmation_age_ms"] is None


def test_duplicate_detections_do_not_confirm_two_targets():
    truth = [dict(label="red_balloon", box=[0, 0, 10, 10])]
    detection = dict(label="red_balloon", box=[0, 0, 10, 10], score=.9, accepted=True)
    assert matched_targets(truth, [detection, detection]) == ({0}, 1)


@pytest.mark.parametrize("cost", [0, -1, 1.2, float("nan"), True])
def test_bad_costs_rejected(cost):
    with pytest.raises(ValueError):
        dispatch("recovery", dict(mser=cost, pixel2=10))
