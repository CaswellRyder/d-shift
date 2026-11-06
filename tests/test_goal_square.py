import cv2
import numpy as np
import pytest

from dtr.goal_square import DEFAULT_MAX_CROPS, find_goals, find_squares, goal_candidates, square_candidates


class FakePredictor:
    metadata = {
        "task": "goal",
        "classes": ["background", "orange_circle", "orange_square", "orange_triangle",
                    "yellow_circle", "yellow_square", "yellow_triangle"],
    }

    def __init__(self, orange, yellow):
        self.scores = [1 - orange - yellow, 0, orange, 0, 0, yellow, 0]

    def predict(self, crop):
        assert crop.size > 0
        return {"scores": self.scores}


def square_frame(color):
    # OpenCV hue 6 is in the orange band; hue 30 is in the disjoint yellow band.
    hue = {"orange": 6, "yellow": 30}[color]
    rgb_color = cv2.cvtColor(np.uint8([[[hue, 220, 230]]]), cv2.COLOR_HSV2RGB)[0, 0]
    rgb = np.full((240, 320, 3), 128, np.uint8)
    cv2.rectangle(rgb, (110, 70), (210, 170), tuple(map(int, rgb_color)), 8)
    candidates = square_candidates(rgb)
    assert len(candidates) == 1 and candidates[0]["color"] == color
    assert candidates[0]["hollow"] > 0.5
    return rgb


@pytest.mark.parametrize("mask_color", ["orange", "yellow"])
def test_ambiguous_goal_color_is_not_a_detection(mask_color):
    result = find_squares(square_frame(mask_color), FakePredictor(orange=0.44, yellow=0.46))
    assert result["neural_calls"] == 1
    assert result["detections"] == []


def test_confident_goal_color_reports_conditional_color_and_shape_scores():
    result = find_squares(square_frame("orange"), FakePredictor(orange=0.85, yellow=0.05))
    (det,) = result["detections"]
    assert det["label"] == "orange_square" and det["color"] == "orange"
    assert det["score"] == pytest.approx(0.90)
    assert det["color_score"] == pytest.approx(0.85 / 0.90)


def test_model_color_must_agree_with_goal_mask():
    result = find_squares(square_frame("orange"), FakePredictor(orange=0.0, yellow=0.9))
    assert result["detections"] == []


def test_color_threshold_is_independent_of_shape_threshold():
    rgb = square_frame("orange")
    predictor = FakePredictor(orange=0.72, yellow=0.18)
    (det,) = find_squares(rgb, predictor, color_threshold=0.8)["detections"]
    assert det["color_score"] == pytest.approx(0.8)
    assert find_squares(rgb, predictor, color_threshold=0.81)["detections"] == []
    assert find_squares(rgb, FakePredictor(0.7, 0), color_threshold=0.5)["detections"] == []


def shape_frame(shape, color):
    rgb = np.full((240, 320, 3), 128, np.uint8)
    hue = {"orange": 6, "yellow": 30}[color]
    ink = tuple(map(int, cv2.cvtColor(np.uint8([[[hue, 220, 230]]]), cv2.COLOR_HSV2RGB)[0, 0]))
    if shape == "circle":
        cv2.circle(rgb, (160, 120), 50, ink, 7)
    elif shape == "square":
        cv2.rectangle(rgb, (110, 70), (210, 170), ink, 7)
    else:
        cv2.polylines(rgb, [np.array([[102, 70], [218, 70], [160, 170]])], True, ink, 7)
    return rgb


class ShapePredictor:
    metadata = FakePredictor.metadata

    def __init__(self, label):
        self.scores = [.05] + [0.] * 6
        self.scores[self.metadata["classes"].index(label)] = .95

    def predict_many(self, crops):
        return [{"scores": self.scores} for _ in crops]


@pytest.mark.parametrize("shape", ["circle", "square", "triangle"])
@pytest.mark.parametrize("color", ["orange", "yellow"])
def test_all_six_hoops(shape, color):
    rgb = shape_frame(shape, color)
    (candidate,) = goal_candidates(rgb)
    assert candidate["shape"] == shape and candidate["color"] == color
    assert candidate["hollow"] > .4
    x1, y1, x2, y2 = candidate["box"]
    pad = int(max(x2-x1, y2-y1) * .12)
    assert candidate["crop_box"] == [x1-pad, y1-pad, x2+pad, y2+pad]
    result = find_goals(rgb, ShapePredictor(f"{color}_{shape}"))
    (detection,) = result["detections"]
    assert detection["label"] == f"{color}_{shape}"
    assert detection["center_normalized"] == pytest.approx([0, 0], abs=.01)
    assert result["neural_calls"] == 1


@pytest.mark.parametrize("geometry,model", [(a, b) for a in ("circle", "square", "triangle")
                                           for b in ("circle", "square", "triangle") if a != b])
def test_model_shape_must_match_geometry(geometry, model):
    assert not find_goals(shape_frame(geometry, "yellow"),
                          ShapePredictor(f"yellow_{model}"))["detections"]


@pytest.mark.parametrize("points", [
    [[100, 90], [130, 90], [150, 150], [80, 150]],  # Trapezoid fragment.
    [[160, 60], [217, 101], [195, 168], [125, 168], [103, 101]],  # Pentagon.
    [[100, 70], [210, 175], [205, 180], [95, 75]],  # Thin diagonal strip.
    [[60, 100], [250, 100], [250, 135], [60, 135]],  # Wide rectangle.
])
def test_unmatched_contours_are_rejected_before_inference(points):
    rgb = np.full((240, 320, 3), 128, np.uint8)
    cv2.fillPoly(rgb, [np.array(points)], (230, 220, 30))
    assert goal_candidates(rgb) == []
    assert find_goals(rgb, ShapePredictor("yellow_square"))["neural_calls"] == 0


def test_hollow_then_large_order_and_four_crop_budget():
    rgb = np.full((240, 320, 3), 128, np.uint8)
    for x in (30, 80, 130, 180, 230):
        cv2.rectangle(rgb, (x, 30), (x+25, 55), (230, 220, 30), 3)
    cv2.rectangle(rgb, (90, 110), (190, 210), (230, 220, 30), -1)
    candidates = goal_candidates(rgb)
    assert len(candidates) == 6
    assert all(c["hollow"] > .02 for c in candidates[:5])
    assert candidates[-1]["hollow"] == 0
    result = find_goals(rgb, ShapePredictor("yellow_square"))
    assert result["neural_calls"] == DEFAULT_MAX_CROPS == 4
    assert len(result["detections"]) == 4
    assert find_goals(rgb, ShapePredictor("yellow_square"), max_crops=0)["neural_calls"] == 0


def test_model_taxonomy_must_cover_all_shapes():
    predictor = ShapePredictor("yellow_square")
    predictor.metadata = dict(task="goal", classes=["background", "orange_square", "yellow_square"])
    with pytest.raises(ValueError, match="seven-class"):
        find_goals(shape_frame("square", "yellow"), predictor)


def test_background_mass_and_requested_colour_filter():
    rgb = shape_frame("triangle", "yellow")
    assert goal_candidates(rgb, ("orange",)) == []
    predictor = ShapePredictor("yellow_triangle")
    predictor.scores = [.3, 0, 0, 0, 0, 0, .7]
    assert not find_goals(rgb, predictor)["detections"]
    for bad in [(), ("green",)]:
        with pytest.raises(ValueError):
            goal_candidates(rgb, bad)


@pytest.mark.parametrize('shape', ['circle', 'square', 'triangle'])
def test_joint_classes_are_marginalized_and_colour_threshold_is_independent(shape):
    predictor = ShapePredictor(f'yellow_{shape}')
    predictor.scores = [.1] + [0.] * 6
    predictor.scores[predictor.metadata['classes'].index(f'yellow_{shape}')] = .72
    predictor.scores[predictor.metadata['classes'].index(f'orange_{shape}')] = .18
    rgb = shape_frame(shape, 'yellow')
    (det,) = find_goals(rgb, predictor, threshold=.85, color_threshold=.8)['detections']
    assert det['score'] == pytest.approx(.9)
    assert det['color_score'] == pytest.approx(.8)
    assert not find_goals(rgb, predictor, color_threshold=.81)['detections']


def test_known_false_positive_fragment_is_never_a_square_hint():
    from dtr.goal_square import shape_hint
    hull = np.array([[153, 142], [151, 146], [125, 144], [134, 128], [143, 128]], np.int32)
    # A broken-off piece of a real triangle may still hint triangle, but never square;
    # the model called it a square, so the shape-agreement check rejects it.
    assert shape_hint(hull)[0] != "square"


def ink(color):
    hue = {"orange": 6, "yellow": 30}[color]
    return tuple(map(int, cv2.cvtColor(np.uint8([[[hue, 220, 230]]]), cv2.COLOR_HSV2RGB)[0, 0]))


def test_crop_budget_takes_the_nearest_goal_not_the_largest_contour():
    rgb = np.full((240, 320, 3), 128, np.uint8)
    # The 88 px circle (1.13 m) is nearer than the 116 px-wide triangle (1.613 m),
    # even though the triangle's contour is larger.
    cv2.circle(rgb, (70, 120), 44, ink("orange"), 6)
    cv2.polylines(rgb, [np.array([[150, 70], [266, 70], [208, 170]])], True, ink("orange"), 6)
    candidates = goal_candidates(rgb)
    assert [c["shape"] for c in candidates] == ["circle", "triangle"]
    result = find_goals(rgb, ShapePredictor("orange_circle"), max_crops=1)
    assert [d["label"] for d in result["detections"]] == ["orange_circle"]


def test_crop_budget_alternates_colours():
    rgb = np.full((240, 320, 3), 128, np.uint8)
    for x in (20, 90, 160):
        cv2.rectangle(rgb, (x, 30), (x + 50, 80), ink("orange"), 4)
    cv2.rectangle(rgb, (100, 140), (130, 170), ink("yellow"), 4)
    assert [c["color"] for c in goal_candidates(rgb)[:2]] == ["orange", "yellow"]


def test_small_round_blob_lets_the_model_choose_circle_or_square():
    rgb = np.full((240, 320, 3), 128, np.uint8)
    cv2.circle(rgb, (160, 120), 4, ink("orange"), 1)
    candidates = goal_candidates(rgb)
    assert len(candidates) == 1 and set(candidates[0]["shapes"]) == {"circle", "square"}
    labels = [d["label"] for d in find_goals(rgb, ShapePredictor("orange_circle"))["detections"]]
    assert labels == ["orange_circle"]
    assert not find_goals(rgb, ShapePredictor("orange_triangle"))["detections"]
