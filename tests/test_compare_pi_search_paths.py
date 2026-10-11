import pytest

from scripts.compare_pi_search_paths import summarize, validate_identity


def test_processing_and_live_rates_are_distinct():
    rows = [dict(processing_ms=100, search_ms=70, inference_ms=30, neural_calls=2,
                 sensor_to_result_ms=200, timestamp_valid=True,
                 sensor_timestamp_increasing=True)] * 4
    result = summarize(rows, elapsed=0.8)
    assert result["processing_fps"] == 10
    assert result["live_fps"] == 5
    assert result["valid_timestamps"] == 4
    assert result["sensor_to_result_ms"]["p95"] == 200


@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf")])
def test_bad_duration_rejected(duration):
    with pytest.raises(ValueError):
        summarize([dict(processing_ms=duration, neural_calls=0)])


def test_invalid_sensor_does_not_become_fresh():
    result = summarize([dict(processing_ms=1, neural_calls=0, sensor_to_result_ms=None,
                             timestamp_valid=False, sensor_timestamp_increasing=False)], 1)
    assert "sensor_to_result_ms" not in result
    assert result["valid_timestamps"] == 0
    assert not result["all_sensor_timestamps_increasing"]


def test_identity_requires_exact_source_and_model():
    validate_identity({"model": "a", "source": "b"}, {"model": "a", "source": "b"})
    with pytest.raises(ValueError):
        validate_identity({"model": "a"}, {"model": "c"})


@pytest.fixture
def replay_fixture(tmp_path):
    from dtr.data import sha256, write_json
    base, replay = tmp_path / "base", tmp_path / "replay"
    base.mkdir()
    replay.mkdir()
    write_json(base / "inputs.json", dict(frames=[dict(panel="development", truth=[])] * 2))
    identity = dict(inputs_sha256=sha256(base / "inputs.json"))
    golden = tmp_path / "golden.json"
    write_json(golden, dict(identity=identity,
                           predictions={m: [[], []] for m in ("mser", "pixel1", "pixel2")}))
    trials = []
    for i, method in enumerate(("mser", "pixel1", "pixel2", "pixel2", "pixel1", "mser")):
        rows = [dict(index=j, processing_ms=10, neural_calls=0, detections=[]) for j in range(2)]
        path = replay / f"{i}-{method}.json"
        write_json(path, dict(rows=rows))
        trials.append(dict(trial=i, method=method, frames=2, rows_sha256=sha256(path),
                           elapsed_seconds=None, summary=summarize(rows)))
    report = dict(identity=identity, trials=trials, machine="armv6l", mode="replay",
                  golden_sha256=sha256(golden), deployment_approved=False, test_evaluated=False,
                  flight_commands=None, runtime_library_sha256="runtime", region_library_sha256="region")
    write_json(replay / "report.json", report)
    return base, golden, replay


def test_review_counts_unique_frames_not_repetitions(replay_fixture):
    from scripts.review_pi_search_paths import review
    result = review(*replay_fixture, [])
    assert result["unique_development_frames"] == 2
    assert result["replay"]["mser"]["processing_fps"] == 100


def test_review_rejects_tampered_raw_rows(replay_fixture):
    from dtr.data import write_json
    from scripts.review_pi_search_paths import review
    write_json(replay_fixture[2] / "0-mser.json", dict(rows=[]))
    with pytest.raises(ValueError, match="Changed raw rows"):
        review(*replay_fixture, [])
