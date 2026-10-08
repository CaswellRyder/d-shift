"""Engineering fixtures only: colored patches are not balloon accuracy evidence."""
import io

import cv2
import numpy as np
import pytest
from PIL import Image

from dtr.detail_vision import observe_detail
from dtr.temporal import TemporalVision
from dtr.vision import RED_BLUE_PROFILE, color_masks, observe, proposals, validate_model_profile
from dtr.web import VisionService


class RedBluePredictor:
    metadata = dict(task="balloon", classes=["background", "red_balloon", "blue_balloon"],
                    synthetic_training=True, threshold=.8, deployment_approved=False)

    def __init__(self, *args):
        self.calls = 0

    def predict(self, rgb):
        self.calls += 1
        label = "red_balloon" if rgb[:, :, 0].sum() > rgb[:, :, 2].sum() else "blue_balloon"
        return dict(label=label, score=.95, accepted=True)


def scene():
    rgb = np.zeros((240, 320, 3), np.uint8)
    rgb[30:50, 30:50] = (230, 0, 0)
    rgb[30:50, 80:100] = (0, 0, 230)
    return rgb


def test_masks_cover_red_wrap_blue_and_reject_obvious_other_colors():
    hues = [0, 5, 175, 179, 100, 120, 60, 150, 30, 20]
    hsv = np.array([[[h, 220, 200] for h in hues]], np.uint8)
    red, blue = color_masks(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), "balloon", RED_BLUE_PROFILE)
    assert red.tolist() == [[255] * 4 + [0] * 6]
    assert blue.tolist() == [[0] * 4 + [255] * 2 + [0] * 4]
    for value in (0, 20, 200, 255):
        assert not any(m.any() for m in color_masks(
            np.full((8, 8, 3), value, np.uint8), "balloon", RED_BLUE_PROFILE))
    with pytest.raises(ValueError, match="only valid"):
        color_masks(scene(), "goal", RED_BLUE_PROFILE)


def test_search_preserves_budgets_and_does_not_double_count_highlight_holes():
    rgb = scene()
    rgb[36:42, 36:42] = 255
    found = proposals(rgb, "balloon", profile=RED_BLUE_PROFILE)
    assert len(found) == 2
    assert {r["color_group"] for r in found} == {0, 1}
    assert len(proposals(rgb, "balloon", limit=1, profile=RED_BLUE_PROFILE)) == 1
    assert {tuple(r["box"]) for r in found} == {(30, 30, 50, 50), (80, 30, 100, 50)}


@pytest.mark.parametrize("metadata", [
    {}, dict(task="goal", classes=RedBluePredictor.metadata["classes"]),
    dict(task="balloon", classes=["background", "green_balloon", "purple_balloon"]),
    dict(task="balloon", classes=["background", "red_balloon"]),
])
def test_mismatched_weights_rejected_before_inference(metadata):
    predictor = RedBluePredictor()
    predictor.metadata = metadata
    for run in (lambda: observe(scene(), predictor, profile=RED_BLUE_PROFILE),
                lambda: observe_detail(scene(), predictor, profile=RED_BLUE_PROFILE),
                lambda: TemporalVision(predictor, proposal_profile=RED_BLUE_PROFILE)):
        with pytest.raises(ValueError, match="red/blue balloon model"):
            run()
    assert predictor.calls == 0


@pytest.mark.parametrize("profile", ["v2", "balloon_components", "goal_lut"])
def test_red_blue_model_cannot_silently_use_legacy_masks(profile):
    with pytest.raises(ValueError, match="require the balloon_red_blue"):
        validate_model_profile(RedBluePredictor.metadata, profile)


def test_static_detail_and_temporal_paths_keep_correct_identity_and_bounds():
    predictor = RedBluePredictor()
    for result in (observe(scene(), predictor, profile=RED_BLUE_PROFILE),
                   observe_detail(scene(), predictor, profile=RED_BLUE_PROFILE)):
        assert {r["label"] for r in result["observations"]} == {"red_balloon", "blue_balloon"}
        assert result["accepted_count"] == 2
        assert result["flight_commands"] is None and not result["deployment_approved"]
    tracker = TemporalVision(predictor, budget=1, proposal_profile=RED_BLUE_PROFILE)
    seen = set()
    for index in range(3):
        result = tracker.observe(scene(), now=index*.1)
        seen.update(r["label"] for r in result["observations"])
        assert result["inference_calls"] <= 1
    assert seen == {"red_balloon", "blue_balloon"}
    assert not tracker.observe(np.zeros_like(scene()), now=.3)["observations"]


def test_viewer_selects_balloon_profile_without_changing_goal_profile(monkeypatch):
    def factory(path, allow_unvalidated):
        predictor = RedBluePredictor()
        if path == "goal":
            predictor.metadata = dict(predictor.metadata, task="goal",
                                      classes=["background", "orange_square", "yellow_square"])
        return predictor

    monkeypatch.setattr("dtr.web.Predictor", factory)
    service = VisionService({"balloon": "balloon", "goal": "goal"}, allow_unvalidated=True)
    assert service.profiles == {"balloon": RED_BLUE_PROFILE, "goal": "balloon_components"}
    assert service.config()["tasks"]["balloon"]["proposal_profile"] == RED_BLUE_PROFILE
    stream = io.BytesIO()
    Image.fromarray(scene()).save(stream, format="JPEG")
    result = service.process(stream.getvalue(), "balloon", "test", 0)
    assert result["proposal_profile"] == RED_BLUE_PROFILE
    assert {r["label"] for r in result["observations"]} == {"red_balloon", "blue_balloon"}


def test_legacy_balloon_profile_still_finds_green_and_purple():
    rgb = scene()
    rgb[30:50, 30:50] = (0, 230, 0)
    rgb[30:50, 80:100] = (230, 0, 230)
    assert len(proposals(rgb, "balloon", profile="balloon_components")) == 2
    assert not proposals(rgb, "balloon", profile=RED_BLUE_PROFILE)
