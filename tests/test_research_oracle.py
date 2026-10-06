import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "research_oracle", Path(__file__).resolve().parents[1] / "scripts/research_oracle_crops.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_oracle_clips_with_context_without_empty_rounding():
    rgb = np.zeros((24, 32, 3), np.uint8)
    assert MODULE.oracle_crop(rgb, [.2, .2, 2.5, 3.5]).shape == (4, 3, 3)
    assert MODULE.oracle_crop(rgb, [30.5, 22.5, 32, 24]).shape == (2, 2, 3)
    for box in ([1, 2, 0, 3], [40, 40, 45, 45], [0, 0, float("nan"), 10]):
        with pytest.raises(ValueError):
            MODULE.oracle_crop(rgb, box)


def test_size_bins_boundaries():
    assert [MODULE.size_bin([0,0,i,i]) for i in (7,8,15,16,31,32)] == [
        "lt8", "8to16", "8to16", "16to32", "16to32", "ge32"]


def test_oracle_accuracy_separate_from_acceptance_and_coverage():
    rows = [dict(model="m", resolution="320x240", truth="orange_circle", size_bin="lt8",
                 prediction=dict(label="orange_circle", accepted=False),
                 covered12=False, covered64=True)]
    all_result = next(r for r in MODULE.summarize(rows) if r["category"] == "all")
    assert all_result["accuracy"] == 1
    assert all_result["accepted_correct_rate"] == 0
    assert all_result["covered12"] == 0
    assert all_result["covered64"] == 1
