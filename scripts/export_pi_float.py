"""Export the frozen student as FP32 for an actual-device runtime comparison."""
import argparse
from pathlib import Path

import keras
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

from dtr.data import read_json, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", choices=("student.keras", "latest.keras"),
                        default="student.keras", help="Selected best or fixed final epoch; never overwrites weights")
    args = parser.parse_args()
    run, output = Path(args.run), Path(args.output)
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(output)
    checkpoint = run / args.checkpoint
    model = keras.models.load_model(checkpoint, compile=False)
    original = read_json(run / "student.int8.json")
    size = original["size"]

    @tf.function(input_signature=[tf.TensorSpec([1,size,size,3], tf.float32)])
    def serve(x):
        return model(x, training=False)

    frozen = convert_variables_to_constants_v2(serve.get_concrete_function())
    converter = tf.lite.TFLiteConverter.from_concrete_functions([frozen])
    binary = converter.convert()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(binary)
    meta = {**original, "sha256": sha256(output), "bytes": len(binary), "tensor_dtype": "float32",
            "integer_only_verified": False, "input_quantization": None, "output_quantization": None,
            "calibration_samples": 0, "calibration_split": None, "output": "float logits; softmax",
            "source_student_sha256": sha256(checkpoint), "source_checkpoint": args.checkpoint,
            "deployment_approved": False, "pi_zero_verified": False}
    write_json(output.with_suffix(".json"), meta)
    print(dict(path=str(output), bytes=len(binary)))


if __name__ == "__main__":
    main()
