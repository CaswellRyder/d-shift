"""Descriptive error partitions at a fixed operating point, not inferred label corrections."""

from collections import Counter

from .detector_metrics import match, size_bin, summarize
from .tracking import iou


def diagnose(frames, standard):
    metrics = summarize(frames, standard)  # Validate inputs using the acceptance evaluator.
    classes = {name: dict(missed=Counter(), false_positive=Counter(), missed_by_size=Counter())
               for name in standard["classes"]}
    confusion = Counter()
    threshold = standard["iou_threshold"]
    for frame in frames:
        truth = frame["truth"]
        predictions = [p for p in frame["predictions"]
                       if p["score"] >= standard["confidence_threshold"]]
        hits = match(truth, predictions, threshold)
        used = {index for _, index in hits if index is not None}
        for index, target in enumerate(truth):
            if index in used:
                continue
            overlaps = [(p, iou(p["box"], target["box"])) for p in predictions]
            same = max((v for p, v in overlaps if p["label"] == target["label"]), default=0)
            other = max(((p, v) for p, v in overlaps if p["label"] != target["label"]),
                        key=lambda pair: pair[1], default=(None, 0))
            if same >= threshold:
                reason = "same_class_box_claimed_by_another_target"
            elif other[1] >= threshold:
                reason = "wrong_class_at_matching_iou"
                confusion[f"{target['label']} -> {other[0]['label']}"] += 1
            elif same >= .1:
                reason = "same_class_localization_below_iou"
            elif other[1] >= .1:
                reason = "wrong_class_and_localization"
            else:
                reason = "no_overlapping_prediction_at_fixed_confidence"
            row = classes[target["label"]]
            row["missed"][reason] += 1
            row["missed_by_size"][size_bin(target["box"], *frame["image_size"])] += 1
        for prediction, index in hits:
            if index is not None:
                continue
            overlaps = [(t, iou(t["box"], prediction["box"])) for t in truth]
            same = max((v for t, v in overlaps if t["label"] == prediction["label"]), default=0)
            other = max((v for t, v in overlaps if t["label"] != prediction["label"]), default=0)
            if same >= threshold:
                reason = "duplicate_at_matching_iou"
            elif other >= threshold:
                reason = "wrong_class_at_matching_iou"
            elif same >= .1:
                reason = "same_class_localization_below_iou"
            elif other >= .1:
                reason = "wrong_class_and_localization"
            else:
                reason = "no_labeled_overlap"
            classes[prediction["label"]]["false_positive"][reason] += 1
    for name, row in classes.items():
        expected = metrics["classes"][name]
        assert sum(row["missed"].values()) == expected["targets"] - expected["true_positive"]
        assert sum(row["false_positive"].values()) == expected["false_positive"]
    return dict(classes=classes, missed_target_confusions=dict(confusion),
                scope="Descriptive overlaps with supplied labels; not error causes or corrected labels",
                loose_overlap_iou=.1, training_approved=False, label_corrections=False)
