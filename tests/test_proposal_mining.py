import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "mining", Path(__file__).parents[1] / "scripts/mine_dtr_proposals.py"
)
mining = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mining)

review_spec = importlib.util.spec_from_file_location(
    "review", Path(__file__).parents[1] / "scripts/admit_proposal_review.py"
)
review = importlib.util.module_from_spec(review_spec)
review_spec.loader.exec_module(review)


def test_labels_come_from_boxes_not_classifier_predictions():
    candidate = {"box": [10, 10, 30, 30], "crop_box": [8, 8, 32, 32]}
    assert mining.assign(candidate, [("green_balloon", [10, 10, 30, 30])]) == "green_balloon"
    assert mining.assign(candidate, [(None, [10, 10, 30, 30])]) is None
    assert mining.assign(candidate, [("green_balloon", [60, 60, 80, 80])]) == "background"
    assert mining.assign(candidate, []) == "background"


def test_partial_and_padded_crop_overlaps_are_not_negative_labels():
    candidate = {"box": [10, 10, 30, 30], "crop_box": [8, 8, 32, 32]}
    assert mining.assign(candidate, [("green_balloon", [20, 20, 50, 50])]) is None
    assert mining.assign(candidate, [("green_balloon", [31, 20, 50, 50])]) is None


def test_review_admits_only_explicit_training_negatives():
    queue = [dict(label="background", split="train", path=str(i)) for i in range(3)]
    assert review.select_reviewed(queue, {"admit": [2]}) == [queue[2]]
    for indices in ([0, 0], [3], [-1]):
        with pytest.raises(ValueError, match="indices"):
            review.select_reviewed(queue, {"admit": indices})
    queue[1]["split"] = "val"
    queue[2]["label"] = "green_balloon"
    for indices in ([1], [2]):
        with pytest.raises(ValueError, match="training negatives"):
            review.select_reviewed(queue, {"admit": indices})
