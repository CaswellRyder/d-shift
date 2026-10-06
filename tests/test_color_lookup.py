import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest

from dtr.color_lookup import GoalColorLookup
from dtr.vision import color_masks, proposals
from dtr.data import write_json, sha256
from dtr.temporal import TemporalVision


@pytest.fixture(scope="module")
def lookup(tmp_path_factory):
    root = tmp_path_factory.mktemp("lookup")
    path = root/"colors.bin"
    spec = importlib.util.spec_from_file_location("build_lookup",Path(__file__).parents[1]/"scripts/build_goal_lookup.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    values = np.empty((256,256,256),np.uint8)
    for red in range(256):
        a,b = color_masks(builder.rgb_slice(red),"goal","balloon_components")
        values[red] = (a//255)|((b//255)<<1)
    path.write_bytes(values.tobytes())
    write_json(path.with_suffix(".json"),dict(contract="dtr-goal-rgb24-hsv-v1",sha256=sha256(path)))
    return path,GoalColorLookup(path)


def test_lookup_matches_masks_random_and_strided(lookup):
    _,lut = lookup
    rng = np.random.default_rng(19)
    for rgb in [rng.integers(0,256,(240,320,3),dtype=np.uint8),
                rng.integers(0,256,(50,80,3),dtype=np.uint8)[:,::2]]:
        before = rgb.copy()
        for got,wanted in zip(lut.masks(rgb),color_masks(rgb,"goal","balloon_components")):
            np.testing.assert_array_equal(got,wanted)
        np.testing.assert_array_equal(rgb,before)
    with pytest.raises(ValueError,match="uint8"):
        lut.masks(np.zeros((10,10,3),np.float32))


def test_opt_in_preserves_proposal_order_and_balloon_path(lookup,monkeypatch):
    path,_ = lookup
    monkeypatch.setenv("DTR_GOAL_COLOR_LOOKUP",str(path))
    rgb = np.zeros((240,320,3),np.uint8)
    cv2.rectangle(rgb,(20,20),(80,80),(250,40,20),2)
    cv2.circle(rgb,(160,80),20,(250,240,10),2)
    cv2.circle(rgb,(180,160),12,(30,230,30),-1)
    for task in ("goal","balloon"):
        assert proposals(rgb,task,profile="goal_lut") == proposals(rgb,task,profile="balloon_components")


def test_lookup_requires_explicit_valid_configuration(tmp_path,monkeypatch):
    monkeypatch.delenv("DTR_GOAL_COLOR_LOOKUP",raising=False)
    with pytest.raises(ValueError,match="explicit"):
        color_masks(np.zeros((10,10,3),np.uint8),"goal","goal_lut")
    path = tmp_path/"bad.bin"
    path.write_bytes(b"bad")
    write_json(path.with_suffix(".json"),dict(contract="dtr-goal-rgb24-hsv-v1",sha256=sha256(path)))
    with pytest.raises(ValueError,match="size/contract/hash"):
        GoalColorLookup(path)
    with pytest.raises(ValueError,match="proposal profile"):
        TemporalVision(None,proposal_profile="unknown")
