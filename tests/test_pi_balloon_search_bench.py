import copy

import numpy as np
import pytest

from scripts import pi_balloon_search_bench as bench
from scripts.summarize_balloon_search_pi import validate_trial


def observation():
    return dict(
        box=[10, 10, 30, 30],
        crop_box=[8, 8, 32, 32],
        label="red_balloon",
        scores=[0.01, 0.98, 0.01],
        accepted=True,
        raw_accepted=True,
        suppressed=False,
    )


def test_golden_parity_accepts_small_numeric_drift_only():
    expected = observation()
    actual = dict(expected, scores=[0.0101, 0.9799, 0.01])
    assert 0 < bench.check_predictions([actual], [expected]) < 0.001
    assert bench.check_predictions([], []) == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("box", [11, 10, 30, 30]),
        ("crop_box", [9, 8, 32, 32]),
        ("label", "blue_balloon"),
        ("accepted", False),
        ("raw_accepted", False),
        ("suppressed", True),
        ("scores", [0.1, 0.89, 0.01]),
        ("scores", [0, 1]),
        ("scores", [float("nan"), 1, 0]),
    ],
)
def test_scene_parity_fails_on_geometry_selection_or_class_drift(field, value):
    expected = observation()
    with pytest.raises(ValueError):
        bench.check_predictions([dict(expected, **{field: value})], [expected])


def test_extra_or_missing_proposals_fail():
    with pytest.raises(ValueError, match="count"):
        bench.check_predictions([], [observation()])


def test_processing_uses_exact_crop_and_single_model(monkeypatch):
    proposal = dict(
        box=[10, 10, 30, 30], crop_box=[8, 8, 32, 32], area=400, color_group=0, proposal_score=10.0
    )
    monkeypatch.setattr(bench, "experimental", lambda rgb, search: [proposal])
    calls = []

    class Predictor:
        def predict(self, rgb):
            calls.append(rgb.shape)
            return dict(label="red_balloon", scores=[0.01, 0.98, 0.01], score=0.98, accepted=True)

    untouched = copy.deepcopy(proposal)
    result = bench.process_frame(np.zeros((240, 320, 3), np.uint8), Predictor(), "mser")
    assert calls == [(24, 24, 3)] and result["neural_calls"] == 1
    assert result["detections"][0]["accepted"] and proposal == untouched
    assert result["processing_ms"] == pytest.approx(
        result["search_ms"] + result["inference_ms"] + result["selection_ms"]
    )
    with pytest.raises(ValueError):
        bench.process_frame(np.zeros((240, 320, 3), np.uint8), Predictor(), "unsupported")


def replay_fixture():
    frame = dict(source="scene.png", panel="indoor", truth=[])
    report = dict(
        machine="armv6l",
        camera_used=False,
        deployment_approved=False,
        test_evaluated=False,
        flight_commands=None,
        unique_frames=1,
        frames=2,
        rounds=2,
    )
    rows = [
        dict(
            round=i,
            index=0,
            source="scene.png",
            panel="indoor",
            detections=[observation()],
            processing_ms=3.0,
            search_ms=1.0,
            inference_ms=1.0,
            selection_ms=1.0,
        )
        for i in range(2)
    ]
    return report, rows, dict(frames=[frame])


def test_repetitions_never_inflate_accuracy_sample_count():
    report, rows, inputs = replay_fixture()
    assert len(validate_trial(report, rows, inputs)) == 1


@pytest.mark.parametrize(
    "fault",
    [
        "hardware",
        "camera",
        "approval",
        "test",
        "rounds",
        "missing",
        "order",
        "source",
        "scores",
        "nan",
    ],
)
def test_invalid_or_incomplete_replay_evidence_fails(fault):
    report, rows, inputs = replay_fixture()
    if fault == "hardware":
        report["machine"] = "arm64"
    elif fault == "camera":
        report["camera_used"] = True
    elif fault == "approval":
        report["deployment_approved"] = True
    elif fault == "test":
        report["test_evaluated"] = True
    elif fault == "rounds":
        report["rounds"] = True
    elif fault == "missing":
        rows.pop()
    elif fault == "order":
        rows[1]["round"] = 0
    elif fault == "source":
        rows[1]["source"] = "different.png"
    elif fault == "scores":
        rows[1]["detections"][0]["scores"] = [0, 1, 0]
    elif fault == "nan":
        rows[1]["processing_ms"] = float("nan")
    with pytest.raises(ValueError):
        validate_trial(report, rows, inputs)
