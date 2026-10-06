"""Offboard Keras teacher inference, using the same RGB crop contract as training."""

import time
from pathlib import Path

import keras
import numpy as np
from PIL import Image

from .data import read_json, sha256
from .runtime import softmax
from .goal_evidence import goal_evidence


class TeacherPredictor:
    def __init__(self, model_path, allow_unvalidated=False):
        path = Path(model_path)
        self.metadata = read_json(path.with_suffix(".json"))
        if self.metadata.get("kind") != "keras_teacher" or sha256(path) != self.metadata["sha256"]:
            raise ValueError("Teacher metadata or checksum mismatch")
        if not allow_unvalidated and not self.metadata.get("deployment_approved", False):
            raise ValueError("Unvalidated teacher: explicitly enable research inference")
        self.model = keras.models.load_model(path, compile=False)
        if self.model.input_shape != (None, self.metadata["size"], self.metadata["size"], 3):
            raise ValueError("Teacher input shape does not match metadata")
        if self.model.output_shape != (None, len(self.metadata["classes"])):
            raise ValueError("Teacher classes do not match metadata")

    def predict(self, rgb):
        return self.predict_many([rgb])[0]

    def _values(self, rgb):
        if isinstance(rgb, (str, Path)):
            with Image.open(rgb) as opened:
                image = opened.convert("RGB")
        else:
            array = np.asarray(rgb)
            if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
                raise ValueError("Expected HWC RGB uint8 image")
            image = Image.fromarray(array)
        size = self.metadata["size"]
        return np.asarray(image.resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32)

    def predict_many(self, images):
        """One bounded offboard batch per frame; latency per result is amortized."""
        if len(images) > 64:
            raise ValueError("At most 64 teacher crops per batch")
        if not len(images):
            return []
        start = time.perf_counter()
        values = np.stack([self._values(image) for image in images])
        logits = self.model(values, training=False).numpy()
        if not np.isfinite(logits).all():
            raise ValueError("Non-finite teacher prediction")
        elapsed = (time.perf_counter() - start) * 1000 / len(images)
        return [self._result(row, elapsed) for row in logits]

    def _result(self, logits, elapsed):
        scores = softmax(logits)
        best = int(scores.argmax())
        label = self.metadata["classes"][best]
        result = {
            "label": label,
            "score": float(scores[best]),
            "scores": scores.tolist(),
            "accepted": bool(label != "background" and scores[best] >= self.metadata["threshold"]),
            "latency_ms": elapsed,
            "latency_kind": "amortized batch preprocessing and inference",
            "synthetic_training": self.metadata["synthetic_training"],
            "deployment_approved": self.metadata["deployment_approved"],
            "engine": "keras_teacher",
        }
        if self.metadata["task"] == "goal":
            result["goal_evidence"] = goal_evidence(scores,self.metadata["classes"])
        return result
