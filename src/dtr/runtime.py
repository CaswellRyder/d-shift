"""CPU crop inference. Imports TensorFlow only if tflite_runtime is unavailable.

No implicit camera capture, network calls, or flight commands.
"""

import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .goal_evidence import goal_evidence


def quantize(values, scale, zero_point, dtype=np.int8):
    if scale <= 0:
        raise ValueError("Invalid quantization scale")
    bounds = np.iinfo(dtype)
    return np.clip(np.rint(values / scale + zero_point), bounds.min, bounds.max).astype(dtype)


def softmax(logits):
    exp = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


class Predictor:
    def __init__(self, model_path, allow_unvalidated=False):
        path = Path(model_path)
        self.metadata = json.loads(path.with_suffix(".json").read_text())
        if hashlib.sha256(path.read_bytes()).hexdigest() != self.metadata["sha256"]:
            raise ValueError("Model checksum does not match metadata")
        if not allow_unvalidated and not self.metadata.get("deployment_approved", False):
            raise ValueError(
                "Unvalidated research model: explicitly enable allow_unvalidated for testing"
            )
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            try:
                import tensorflow as tf
                Interpreter = tf.lite.Interpreter
            except ImportError:
                from .native_tflite import Interpreter
        self.interpreter = Interpreter(model_path=str(path), num_threads=1)
        self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()[0]
        self.output = self.interpreter.get_output_details()[0]
        expected = [1, self.metadata["size"], self.metadata["size"], 3]
        if self.input["shape"].tolist() != expected:
            raise ValueError("Unexpected model input shape")
        encoding = self.metadata.get("tensor_dtype", "int8")
        if encoding not in ("int8", "float32"):
            raise ValueError("Unsupported model tensor dtype")
        expected_dtype = np.dtype(encoding)
        if self.input["dtype"] != expected_dtype or self.output["dtype"] != expected_dtype:
            raise ValueError("Model tensor dtype does not match metadata")
        if self.output["shape"].tolist() != [1, len(self.metadata["classes"])]:
            raise ValueError("Output classes do not match metadata")

    def predict(self, rgb):
        start = time.perf_counter()
        if isinstance(rgb, (str, Path)):
            with Image.open(rgb) as opened:
                im = opened.convert("RGB")
        else:
            array = np.asarray(rgb)
            if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
                raise ValueError("Expected HWC RGB uint8 image")
            im = Image.fromarray(array)
        size = self.metadata["size"]
        values = np.asarray(im.resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32)[
            None
        ]
        if self.input["dtype"] == np.int8:
            scale, zero = self.input["quantization"]
            values = quantize(values, scale, zero)
        self.interpreter.set_tensor(self.input["index"], values)
        self.interpreter.invoke()
        logits = self.interpreter.get_tensor(self.output["index"])[0].astype(np.float32)
        if self.output["dtype"] == np.int8:
            scale, zero = self.output["quantization"]
            logits = (logits - zero) * scale
        scores = softmax(logits)
        best = int(scores.argmax())
        label = self.metadata["classes"][best]
        result = {
            "label": label,
            "score": float(scores[best]),
            "scores": scores.tolist(),
            "accepted": bool(label != "background" and scores[best] >= self.metadata["threshold"]),
            "latency_ms": (time.perf_counter() - start) * 1000,
            "synthetic_training": self.metadata["synthetic_training"],
            "deployment_approved": self.metadata["deployment_approved"],
        }
        if self.metadata["task"] == "goal":
            result["goal_evidence"] = goal_evidence(scores,self.metadata["classes"])
        result["latency_ms"] = (time.perf_counter()-start)*1000
        return result


def benchmark(model, image, count=100, allow_unvalidated=False):
    if count < 1:
        raise ValueError("Benchmark count must be positive")
    predictor = Predictor(model, allow_unvalidated=allow_unvalidated)
    with Image.open(image) as opened:
        rgb = np.array(opened.convert("RGB"))
    for _ in range(5):
        predictor.predict(rgb)
    times = [predictor.predict(rgb)["latency_ms"] for _ in range(count)]
    return {
        "host": platform.node(),
        "machine": platform.machine(),
        "samples": count,
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
        "max_ms": max(times),
        "scope": "crop resize + quantize + invoke + softmax; not camera loop",
        "armv6_runtime_exercised": platform.machine() == "armv6l",
        "pi_zero_verified": False,
    }
