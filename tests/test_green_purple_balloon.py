"""Engineering fixtures only: colored patches are not balloon accuracy evidence."""
import io

import cv2
import numpy as np
import pytest
from PIL import Image

from dtr.detail_vision import observe_detail
from dtr.temporal import TemporalVision
from dtr.vision import (
    GREEN_PURPLE_PROFILE, color_masks, observe, proposals, validate_model_profile,
)
from dtr.web import VisionService


class GreenPurplePredictor:
    metadata = dict(task="balloon", classes=["background", "green_balloon", "purple_balloon"],
                    synthetic_training=True, threshold=.8, deployment_approved=False)

    def __init__(self, *args):
        self.calls = 0

    def predict(self, rgb):
        self.calls += 1
        label = "green_balloon" if rgb[:, :, 1].sum() > rgb[:, :, 2].sum() else "purple_balloon"
        return dict(label=label, score=.95, accepted=True)


def hsv_patch(hue, sat=150, val=120):
    return cv2.cvtColor(np.full((20, 20, 3), (hue, sat, val), np.uint8), cv2.COLOR_HSV2RGB)


def scene(*hues):
    rgb = np.zeros((240, 320, 3), np.uint8)
    for index, hue in enumerate(hues):
        rgb[30:50, 30 + 50 * index:50 + 50 * index] = hsv_patch(hue)
    return rgb


def test_masks_keep_measured_green_purple_and_exclude_red_blue_drones():
    # Measured DTR purple is hue 114-123; EngDes2 blue is 101-110 and red wraps 0/179.
    hues = [40, 60, 114, 120, 123, 150, 0, 5, 176, 105, 110, 20]
    hsv = np.array([[[h, 200, 160] for h in hues]], np.uint8)
    green, purple = color_masks(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), "balloon",
                                GREEN_PURPLE_PROFILE)
    assert green.tolist() == [[255] * 2 + [0] * 10]
    assert purple.tolist() == [[0] * 2 + [255] * 4 + [0] * 6]
    for value in (0, 20, 200, 255):
        assert not any(m.any() for m in color_masks(
            np.full((8, 8, 3), value, np.uint8), "balloon", GREEN_PURPLE_PROFILE))
    with pytest.raises(ValueError, match="only valid"):
        color_masks(scene(40), "goal", GREEN_PURPLE_PROFILE)


def test_search_proposes_targets_but_not_red_or_blue():
    found = proposals(scene(40, 120, 0, 105), "balloon", profile=GREEN_PURPLE_PROFILE)
    assert {tuple(r["box"]) for r in found} == {(30, 30, 50, 50), (80, 30, 100, 50)}
    assert {r["color_group"] for r in found} == {0, 1}
    # The legacy purple band lets a blue drone through; the corrected one does not.
    assert len(proposals(scene(105), "balloon", profile="balloon_components")) == 1
    assert not proposals(scene(105), "balloon", profile=GREEN_PURPLE_PROFILE)


def test_compact_colour_signal_beats_large_dim_clutter_at_four_crops():
    rgb = np.zeros((240, 320, 3), np.uint8)
    # Saturated colour on a filled small balloon carries more evidence than a
    # large, almost black patch whose hue is dominated by sensor noise.
    for x in (5, 65, 125, 185):
        cv2.rectangle(rgb, (x, 5), (x + 45, 50), (32, 40, 32), -1)
    cv2.circle(rgb, (100, 120), 7, (40, 180, 40), -1)
    cv2.ellipse(rgb, (200, 120), (7, 9), 0, 0, 360,
                cv2.cvtColor(np.uint8([[[120, 180, 160]]]), cv2.COLOR_HSV2RGB)[0, 0].tolist(), -1)
    found = proposals(rgb, 'balloon', limit=4, profile=GREEN_PURPLE_PROFILE)
    assert any(c['box'] == [93, 113, 108, 128] for c in found)
    assert any(c['box'] == [193, 111, 208, 130] for c in found)
    assert len(found) <= 4


def test_near_dull_balloon_keeps_a_slot_beside_tiny_saturated_specks():
    rgb = np.zeros((240, 320, 3), np.uint8)
    cv2.ellipse(rgb, (160, 130), (48, 60), 0, 0, 360, (40, 100, 40), -1)
    for x in (10, 30, 50, 70):
        cv2.circle(rgb, (x, 10), 4, (0, 255, 0), -1)
    found = proposals(rgb, 'balloon', limit=4, profile=GREEN_PURPLE_PROFILE)
    # The best-scoring speck still leads; the near balloon takes the reserved slot.
    assert found[0]['box'] != [112, 70, 209, 191]
    assert found[1]['box'] == [112, 70, 209, 191]


@pytest.mark.parametrize('center', [(10, 130), (310, 130), (160, 8), (160, 232)])
@pytest.mark.parametrize('bgr_hsv', [(60, 150, 100), (130, 150, 120)])
def test_frame_clipped_balloon_is_not_rejected_for_aspect(center, bgr_hsv):
    rgb = np.zeros((240, 320, 3), np.uint8)
    color = cv2.cvtColor(np.uint8([[bgr_hsv]]), cv2.COLOR_HSV2RGB)[0, 0].tolist()
    axes = (48, 60) if center[1] == 130 else (60, 48)
    cv2.ellipse(rgb, center, axes, 0, 0, 360, color, -1)
    found = proposals(rgb, 'balloon', limit=4, profile=GREEN_PURPLE_PROFILE)
    assert len(found) == 1
    x1, y1, x2, y2 = found[0]['box']
    assert max((x2 - x1) / (y2 - y1), (y2 - y1) / (x2 - x1)) > 2  # Visible box exceeds 2:1.


def test_unclipped_elongated_component_still_fails_the_aspect_test():
    rgb = np.zeros((240, 320, 3), np.uint8)
    cv2.ellipse(rgb, (160, 120), (20, 50), 0, 0, 360, (40, 180, 40), -1)
    assert not proposals(rgb, 'balloon', profile=GREEN_PURPLE_PROFILE)


def test_balloon_filter_rejects_unresolved_thin_and_ragged_components():
    rgb = np.zeros((240, 320, 3), np.uint8)
    rgb[10:14, 10:15] = hsv_patch(60)[0, 0]  # Four-pixel side, unresolved.
    rgb[30:36, 30:50] = hsv_patch(120)[0, 0]  # Elongated purple fragment.
    # A branched component has a square bounding box but excessive perimeter.
    cv2.rectangle(rgb, (80, 62), (82, 100), (40, 180, 40), -1)
    cv2.rectangle(rgb, (62, 80), (100, 82), (40, 180, 40), -1)
    assert not proposals(rgb, 'balloon', profile=GREEN_PURPLE_PROFILE)
    assert proposals(rgb, 'balloon', profile='balloon_components')  # Other profiles unchanged.


def test_balloon_inside_bright_hollow_rim_keeps_its_crop_slot():
    rgb = np.zeros((240, 320, 3), np.uint8)
    cv2.circle(rgb, (120, 110), 50, (70, 220, 70), 2)
    cv2.circle(rgb, (120, 110), 7, (40, 140, 40), -1)
    found = proposals(rgb, 'balloon', limit=1, profile=GREEN_PURPLE_PROFILE)
    assert len(found) == 1 and found[0]['box'] == [113, 103, 128, 118]


def test_balloon_colour_signal_has_no_new_hard_brightness_cutoff():
    rgb = np.zeros((240, 320, 3), np.uint8)
    cv2.circle(rgb, (120, 110), 7, (10, 30, 10), -1)
    assert proposals(rgb, 'balloon', limit=1, profile=GREEN_PURPLE_PROFILE)


@pytest.mark.parametrize("metadata", [
    {}, dict(task="goal", classes=GreenPurplePredictor.metadata["classes"]),
    dict(task="balloon", classes=["background", "green_balloon"]),
])
def test_mismatched_weights_rejected_before_inference(metadata):
    predictor = GreenPurplePredictor()
    predictor.metadata = metadata
    for run in (lambda: observe(scene(40), predictor, profile=GREEN_PURPLE_PROFILE),
                lambda: observe_detail(scene(40), predictor, profile=GREEN_PURPLE_PROFILE),
                lambda: TemporalVision(predictor, proposal_profile=GREEN_PURPLE_PROFILE)):
        with pytest.raises(ValueError, match="green/purple balloon model"):
            run()
    assert predictor.calls == 0


def test_red_blue_model_cannot_use_green_purple_masks():
    metadata = dict(task="balloon", classes=["background", "red_balloon", "blue_balloon"])
    with pytest.raises(ValueError, match="require the balloon_red_blue"):
        validate_model_profile(metadata, GREEN_PURPLE_PROFILE)


def test_static_detail_and_temporal_paths_report_targets():
    predictor = GreenPurplePredictor()
    for result in (observe(scene(40, 120, 105), predictor, profile=GREEN_PURPLE_PROFILE),
                   observe_detail(scene(40, 120, 105), predictor, profile=GREEN_PURPLE_PROFILE)):
        assert {r["label"] for r in result["observations"]} == {"green_balloon", "purple_balloon"}
        assert result["accepted_count"] == 2
        assert result["flight_commands"] is None and not result["deployment_approved"]
    tracker = TemporalVision(predictor, budget=1, proposal_profile=GREEN_PURPLE_PROFILE)
    seen = set()
    for index in range(3):
        result = tracker.observe(scene(40, 120), now=index * .1)
        seen.update(r["label"] for r in result["observations"])
        assert result["inference_calls"] <= 1
    assert seen == {"green_balloon", "purple_balloon"}


def test_viewer_uses_green_purple_profile_for_target_models(monkeypatch):
    def factory(path, allow_unvalidated):
        predictor = GreenPurplePredictor()
        if path == "goal":
            predictor.metadata = dict(predictor.metadata, task="goal",
                                      classes=["background", "orange_square", "yellow_square"])
        return predictor

    monkeypatch.setattr("dtr.web.Predictor", factory)
    service = VisionService({"balloon": "balloon", "goal": "goal"}, allow_unvalidated=True)
    assert service.profiles == {"balloon": GREEN_PURPLE_PROFILE, "goal": "balloon_components"}
    stream = io.BytesIO()
    Image.fromarray(scene(40, 105)).save(stream, format="JPEG", quality=95)
    result = service.process(stream.getvalue(), "balloon", "test", 0)
    assert result["proposal_profile"] == GREEN_PURPLE_PROFILE
    assert [r["label"] for r in result["observations"]] == ["green_balloon"]
