import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "verify_detector", Path(__file__).parents[1] / "scripts/verify_detector_candidates.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_verifier_gates_relabels_and_suppresses_new_same_class_duplicates():
    predictions = [dict(label="yellow_square", box=[0, 0, 10, 10], score=.9),
                   dict(label="yellow_circle", box=[0, 0, 10, 10], score=.8),
                   dict(label="orange_circle", box=[20, 0, 30, 10], score=.99)]
    results = [dict(label="yellow_circle", score=.95, accepted=True),
               dict(label="yellow_circle", score=.99, accepted=True),
               dict(label="background", score=.99, accepted=False)]
    verified = module.verified_predictions(predictions, results)
    assert len(verified) == 1
    assert verified[0]["label"] == "yellow_circle"
    assert verified[0]["detector_label"] == "yellow_square"
    assert verified[0]["score"] == .9  # No confidence rescaling to manufacture a pass.
    assert predictions[0]["label"] == "yellow_square"


def test_verifier_requires_all_candidate_results_and_handles_empty_frames():
    assert module.verified_predictions([], []) == []
    with pytest.raises(ValueError):
        module.verified_predictions([{}], [])


def test_crop_contract_adds_context_and_clips_image_bounds():
    assert module.crop_box([0, 0, 10, 10], 20, 20) == [0, 0, 12, 12]
    assert module.crop_box([10, 10, 20, 20], 20, 20) == [8, 8, 20, 20]
