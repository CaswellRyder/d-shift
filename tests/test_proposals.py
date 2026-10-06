import cv2
import numpy as np
import pytest

from dtr.tracking import iou
from dtr.vision import color_mask, observe, proposals
from dtr.vision import consolidate_rims, orange_local_mask, orange_regions


def test_thin_red_rim_and_blue_purple_are_not_erased():
    rgb = np.zeros((240, 320, 3), np.uint8)
    cv2.rectangle(rgb, (30, 30), (80, 80), (220, 40, 20), 1)
    assert color_mask(rgb, "goal")[30, 50] == 255
    assert any(iou(c["box"], [30, 30, 81, 81]) > 0.8 for c in proposals(rgb, "goal"))
    hsv = np.uint8([[[115, 180, 200]]])
    purple = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0].tolist()
    cv2.ellipse(rgb, (150, 110), (10, 15), 0, 0, 360, purple, -1)
    assert any(iou(c["box"], [140, 95, 161, 126]) > 0.8 for c in proposals(rgb, "balloon"))


def test_bounds_budget_and_balanced_colors():
    rgb = np.zeros((240, 320, 3), np.uint8)
    for x in range(5, 280, 45):
        cv2.rectangle(rgb, (x, 5), (x + 25, 40), (250, 240, 20), 2)
    cv2.circle(rgb, (10, 150), 6, (230, 45, 20), 2)
    found = proposals(rgb, "goal", limit=2)
    assert len(found) == 2 and {r["color_group"] for r in found} == {0, 1}
    for row in found:
        x1, y1, x2, y2 = row["box"]
        a, b, c, d = row["crop_box"]
        assert 0 <= a <= x1 < x2 <= c <= 320
        assert 0 <= b <= y1 < y2 <= d <= 240
    with pytest.raises(ValueError):
        proposals(rgb, "goal", limit=0)
    assert proposals(np.zeros_like(rgb), "balloon") == []


def test_observe_uses_one_batch_and_no_flight_commands():
    class Fake:
        metadata = {"task": "goal"}
        calls = 0

        def predict_many(self, crops):
            self.calls += 1
            return [{"label": "orange_circle", "score": 0.9, "accepted": True} for _ in crops]

    rgb = np.zeros((240, 320, 3), np.uint8)
    cv2.circle(rgb, (80, 80), 20, (240, 40, 20), 2)
    predictor = Fake()
    result = observe(rgb, predictor)
    assert predictor.calls == 1 and result["observations"]
    assert result["flight_commands"] is None and not result["deployment_approved"]
    assert set(result["stage_ms"]) == {"search", "crop_and_classify", "assemble_and_suppress"}
    assert all(np.isfinite(value) and value >= 0 for value in result["stage_ms"].values())
    assert sum(result["stage_ms"].values()) == pytest.approx(result["processing_ms"])


def test_experimental_local_orange_rejects_uniform_warm_background():
    rgb = np.full((240, 320, 3), [150, 140, 130], dtype=np.uint8)
    assert not orange_local_mask(rgb).any()
    cv2.circle(rgb, (80, 80), 18, (180, 160, 130), 2)
    assert orange_local_mask(rgb).any()
    assert any(
        iou(c["box"], [61, 61, 100, 100]) > 0.7
        for c in proposals(rgb, "goal", profile="orange_local")
    )
    assert not proposals(rgb, "goal", profile="v2")
    assert proposals(rgb, "balloon", profile="orange_local") == proposals(
        rgb, "balloon", profile="v2"
    )


def test_conservative_rim_consolidation_preserves_score_and_inputs():
    from copy import deepcopy

    rows = [
        dict(box=[0, 0, 30, 30], color_group=0, proposal_score=1),
        dict(box=[3, 3, 27, 27], color_group=0, proposal_score=4),
        dict(box=[3, 3, 27, 27], color_group=1, proposal_score=5),
        dict(box=[10, 10, 15, 15], color_group=0, proposal_score=2),
    ]
    before = deepcopy(rows)
    found = consolidate_rims(rows, conservative=True)
    assert rows == before
    assert len(found) == 3 and found[0]["box"] == [0, 0, 30, 30]
    assert found[0]["proposal_score"] == 4


def test_balloon_components_exclude_holes_but_preserve_islands_inside_hoops():
    rgb = np.zeros((240, 320, 3), np.uint8)
    # Thick rim keeps the hole below the existing 0.65 proposal IoU merge threshold.
    cv2.circle(rgb, (100, 100), 45, (30, 220, 30), 16)
    cv2.circle(rgb, (100, 100), 9, (30, 220, 30), -1)
    original = proposals(rgb, "balloon", profile="v2")
    components = proposals(rgb, "balloon", profile="balloon_components")
    external = proposals(rgb, "balloon", profile="balloon_external")
    assert len(components) < len(original)
    target = [91, 91, 110, 110]
    assert any(iou(c["box"], target) > 0.8 for c in components)
    assert not any(iou(c["box"], target) > 0.8 for c in external)
    assert proposals(rgb, "goal", profile="balloon_components") == proposals(
        rgb, "goal", profile="v2"
    )


@pytest.mark.parametrize("profile", ["goal_gap9", "orange_regions", "orange_regions_compact"])
def test_goal_experiments_leave_balloon_component_path_unchanged(profile):
    rgb = np.zeros((240, 320, 3), np.uint8)
    cv2.circle(rgb, (100, 100), 45, (30, 220, 30), 16)
    cv2.circle(rgb, (100, 100), 9, (30, 220, 30), -1)
    assert proposals(rgb, "balloon", profile=profile) == proposals(
        rgb, "balloon", profile="balloon_components"
    )


def test_orange_regions_find_faint_small_rim_and_reject_flat_background():
    rgb = np.full((240, 320, 3), [150, 140, 130], dtype=np.uint8)
    assert orange_regions(rgb) == []
    assert orange_regions(np.zeros((2, 2, 3), np.uint8)) == []
    cv2.rectangle(rgb, (60, 60), (74, 72), (180, 160, 130), 2)
    assert not proposals(rgb, "goal", profile="balloon_components")
    found = proposals(rgb, "goal", profile="orange_regions_compact")
    assert any(iou(c["box"], [59, 59, 76, 74]) >= 0.8 for c in found)
    assert len(found) <= 12
    for row in found:
        x1, y1, x2, y2 = row["box"]
        a, b, c, d = row["crop_box"]
        assert 0 <= a <= x1 < x2 <= c <= 320
        assert 0 <= b <= y1 < y2 <= d <= 240


def test_orange_regions_do_not_replace_yellow_geometry_in_simple_fixture():
    rgb = np.zeros((240, 320, 3), np.uint8)
    cv2.rectangle(rgb, (50, 50), (90, 90), (250, 240, 20), 2)
    def yellow(profile):
        return [c for c in proposals(rgb, "goal", profile=profile) if c["color_group"] == 1]

    assert yellow("orange_regions_compact") == yellow("balloon_components")
