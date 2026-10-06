"""Full-integer export with representative training data and artifact contracts."""

from pathlib import Path

import numpy as np
import tensorflow as tf

from .data import SPLITS, load_rgb, sha256, validate, write_json


def export_int8(model, manifest, config, output, provenance, validation_splits=SPLITS):
    output = Path(output)
    if "train" not in validation_splits:
        raise ValueError("Calibration requires validated training data")
    doc = validate(manifest, config, splits=validation_splits)
    rows = [row for row in doc["samples"] if row["split"] == "train"]
    # Randomize deterministically rather than taking the first class in a sorted manifest.
    rng = np.random.default_rng(config["seed"])
    rng.shuffle(rows)
    rows = rows[: config["calibration_samples"]]
    size = config["student_size"]
    root = Path(manifest).resolve().parent

    def representative():
        for row in rows:
            yield [load_rgb(root / row["path"], size)[None]]

    # Pin TF and freeze resource variables before conversion. Direct SavedModel conversion
    # in TF 2.18/Keras 3 can leave READ_VARIABLE nodes that fail during calibration.
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

    @tf.function(input_signature=[tf.TensorSpec([1, size, size, 3], tf.float32)])
    def serve(x):
        return model(x, training=False)

    frozen = convert_variables_to_constants_v2(serve.get_concrete_function())
    converter = tf.lite.TFLiteConverter.from_concrete_functions([frozen])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    binary = converter.convert()
    interpreter = tf.lite.Interpreter(model_content=binary)
    interpreter.allocate_tensors()
    floats = [
        x["name"]
        for x in interpreter.get_tensor_details()
        if np.issubdtype(x["dtype"], np.floating)
    ]
    if floats:
        raise ValueError(f"Float tensors remain: {floats}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(binary)
    inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    metadata = {
        "task": config["task"],
        "classes": config["classes"],
        "size": size,
        "layout": "NHWC",
        "color": "RGB",
        "pixel_range": [0, 255],
        "resize": "Pillow bilinear stretch",
        "normalization": "embedded 1/255",
        "output": "logits; dequantize then softmax",
        "threshold": config["threshold"],
        "threshold_status": "provisional; tune on validation recordings",
        "input_quantization": list(inp["quantization"]),
        "output_quantization": list(out["quantization"]),
        "sha256": sha256(output),
        "bytes": len(binary),
        "calibration_samples": len(rows),
        "calibration_split": "train",
        "integer_only_verified": True,
        "synthetic_training": doc["synthetic"],
        "deployment_approved": False,
        "pi_zero_verified": False,
        "training": provenance,
    }
    write_json(output.with_suffix(".json"), metadata)
    return metadata
