import cv2
import numpy as np
import pytest

from scripts.audit_balloon_search_stages import explain, trace, training_source
from scripts.research_balloon_search import experimental


@pytest.mark.parametrize("stage,reason", [("raw", "no_localized_mser_region"),
    ("contrast", "contrast_filtered"), ("geometry", "geometry_filtered"),
    ("selected", "ranking_or_duplicate_filtered")])
def test_miss_attribution_is_first_stage_that_loses_the_target(stage, reason):
    target = dict(label="blue_balloon", box=[10, 10, 30, 30])
    c = dict(box=target["box"], color_group=1)
    stages = {s: [c] for s in ("raw", "contrast", "geometry", "selected")}
    lost = False
    for s in stages:
        lost |= s == stage
        if lost:
            stages[s] = [dict(c, color_group=0)]
    assert explain(target, stages)["reason"] == reason


def test_instrumented_search_matches_baseline_and_does_not_mutate_pixels():
    rgb = np.full((240, 320, 3), 30, np.uint8)
    cv2.circle(rgb, (90, 90), 30, (10, 20, 200), -1)
    original = rgb.copy()
    stages = trace(rgb)
    assert stages["selected"] == experimental(rgb, "mser")
    assert explain(dict(label="blue_balloon", box=[60, 60, 121, 121]), stages)["reason"] == "covered"
    np.testing.assert_array_equal(rgb, original)


@pytest.mark.parametrize("source", ["train/IMG_111.jpg", "iMg_222.png"])
def test_reserved_family_never_enters_training_audit(source):
    with pytest.raises(ValueError, match="Reserved IMG"):
        training_source(source)


def test_bad_trace_shape_rejected():
    with pytest.raises(ValueError, match="320x240"):
        trace(np.zeros((1, 1, 3), np.uint8))
