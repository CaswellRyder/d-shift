"""Offboard teacher/student scene comparison; not the Pi runtime."""
from pathlib import Path

from dtr.runtime import Predictor


def load_predictor(path):
    if Path(path).suffix == ".keras":
        from dtr.teacher_runtime import TeacherPredictor
        return TeacherPredictor(path, allow_unvalidated=True)
    if Path(path).suffix == ".tflite":
        return Predictor(path, allow_unvalidated=True)
    raise ValueError("Expected .keras teacher or .tflite student")


def classify_candidates(predictor, rgb, candidates):
    crops = [rgb[b:d, a:c] for a, b, c, d in (c["crop_box"] for c in candidates)]
    if predictor.metadata.get("kind") == "keras_teacher":
        predictions = []
        for start in range(0, len(crops), 64):
            predictions.extend(predictor.predict_many(crops[start:start + 64]))
    else:
        predictions = [predictor.predict(crop) for crop in crops]
    return [dict(candidate, **prediction)
            for candidate, prediction in zip(candidates, predictions, strict=True)]
