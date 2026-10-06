import importlib.util
import json
from pathlib import Path

import numpy as np


spec = importlib.util.spec_from_file_location(
    "competition_compare", Path(__file__).parents[1]/"scripts/competition_compare.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def test_scheduler_drops_instead_of_queueing():
    assert bench.next_index(.35, 10, 0) == 3
    assert bench.next_index(.01, 10, 0) == 1
    assert bench.next_index(3.1, 10, 20) == 31


def test_generated_blackout_removes_truth():
    rgb = np.full((240, 320, 3), 120, np.uint8)
    truth = [dict(label="yellow_square", box=[50, 50, 80, 80])]
    frame, labels, phase = bench.scene(rgb, truth, 10)
    assert not frame.any() and labels == [] and phase == "missing"


def test_scene_zero_is_same_input_and_boxes():
    rgb = np.full((240, 320, 3), 120, np.uint8)
    truth = [dict(label="yellow_square", box=[50, 50, 80, 80])]
    frame, labels, phase = bench.scene(rgb, truth, 0)
    assert np.array_equal(frame, rgb)
    assert labels == truth and phase == "present"


def test_scene_return_is_not_blackout():
    rgb = np.full((240, 320, 3), 120, np.uint8)
    truth = [dict(label="yellow_square", box=[50, 50, 80, 80])]
    frame, labels, phase = bench.scene(rgb, truth, 13)
    assert frame.any() and len(labels) == 1 and phase == "present"
    assert json.loads(json.dumps(labels)) == labels
