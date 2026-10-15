import cv2
import numpy as np
import pytest

from dtr.goal_square import find_squares, square_candidates


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
