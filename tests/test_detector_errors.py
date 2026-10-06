from dtr.data import read_json
from dtr.detector_errors import diagnose


def test_error_partitions_preserve_counts_without_claiming_label_corrections():
    standard = read_json("configs/goal-detection-standard.json")
    circle = dict(label="orange_circle", box=[0, 0, 10, 10])
    square = dict(label="orange_square", box=[20, 0, 30, 10])
    triangle = dict(label="orange_triangle", box=[40, 0, 50, 10])
    yellow = dict(label="yellow_circle", box=[60, 0, 70, 10])
    frames = [dict(image_size=[320, 240], truth=[circle, square, triangle, yellow], predictions=[
        dict(circle, score=.9), dict(circle, score=.8),  # One hit, one duplicate.
        dict(square, label="orange_triangle", score=.9),  # Wrong shape.
        dict(triangle, box=[45, 0, 55, 10], score=.8),  # IoU 1/3, bad localization.
        dict(yellow, score=.1),  # Below operating threshold, not a diagnostic match.
        dict(label="yellow_square", box=[100, 0, 110, 10], score=.9),
    ])]
    result = diagnose(frames, standard)
    c = result["classes"]
    assert c["orange_circle"]["false_positive"] == {"duplicate_at_matching_iou": 1}
    assert c["orange_square"]["missed"] == {"wrong_class_at_matching_iou": 1}
    assert c["orange_triangle"]["missed"] == {"same_class_localization_below_iou": 1}
    assert c["yellow_circle"]["missed"] == {"no_overlapping_prediction_at_fixed_confidence": 1}
    assert c["yellow_square"]["false_positive"] == {"no_labeled_overlap": 1}
    assert result["missed_target_confusions"] == {"orange_square -> orange_triangle": 1}
    assert not result["training_approved"] and not result["label_corrections"]


def test_one_box_for_two_targets_is_competition_not_missing_detection():
    standard = read_json("configs/goal-detection-standard.json")
    target = dict(label="orange_circle", box=[0, 0, 10, 10])
    result = diagnose([dict(image_size=[320, 240], truth=[target, target],
                            predictions=[dict(target, score=.9)])], standard)
    assert result["classes"]["orange_circle"]["missed"] == {
        "same_class_box_claimed_by_another_target": 1}
