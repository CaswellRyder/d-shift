import argparse
import json
import os
import platform
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="DTR V4 teachers / tiny INT8 crop students")
    parser.add_argument(
        "--cpu", action="store_true", help="Disable GPU for reproducible diagnostics"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor")
    p = commands.add_parser("pretrained", help="Import public timm V4 weights into native Keras")
    p.add_argument("--output", default="artifacts/pretrained/mobilenetv4_conv_small.keras")
    p.add_argument("--size", type=int, default=96)
    p = commands.add_parser("synthetic", help="Generate clearly marked smoke-test fixtures")
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--per-class", type=int, default=12)
    p = commands.add_parser("prepare-coco")
    p.add_argument("--config", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--sessions")
    p.add_argument("--trust-source-splits", action="store_true")
    p = commands.add_parser("validate")
    p.add_argument("--config", required=True)
    p.add_argument("--manifest", required=True)
    p = commands.add_parser("train", help="Teacher → distillation → INT8 → held-out report")
    p.add_argument("--config", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--pretrained")
    p.add_argument("--smoke", action="store_true")
    p = commands.add_parser("train-teacher", help="Fine-tune V4 only; never starts distillation")
    p.add_argument("--config", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--pretrained")
    p.add_argument(
        "--initial-teacher", help="Verified existing task teacher; retain classifier weights"
    )
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--epoch-budget", type=int)
    p = commands.add_parser("evaluate-teacher", help="Explicit validation or final-test evaluation")
    p.add_argument("--run", required=True)
    p.add_argument("--manifest")
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--output", required=True)
    p = commands.add_parser("predict-teacher")
    p.add_argument("--model", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--allow-unvalidated", action="store_true")
    p = commands.add_parser("predict")
    p.add_argument("--model", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--allow-unvalidated", action="store_true")
    p = commands.add_parser("benchmark")
    p.add_argument("--model", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--count", type=int, default=100)
    p.add_argument("--allow-unvalidated", action="store_true")
    p = commands.add_parser("replay")
    p.add_argument("--model", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-frames", type=int, default=100)
    p.add_argument("--allow-unvalidated", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("KERAS_BACKEND", "tensorflow")
    if args.cpu:
        import tensorflow as tf

        tf.config.set_visible_devices([], "GPU")
    from .data import read_json, validate

    config = read_json(args.config) if hasattr(args, "config") else None
    try:
        if args.command == "doctor":
            import tensorflow as tf
            import keras
            from .models import teacher, student

            a = teacher(3)
            b = student(3)
            tf.debugging.assert_all_finite(a(tf.zeros((1, 96, 96, 3))), "Invalid teacher output")
            result = {
                "python": platform.python_version(),
                "machine": platform.machine(),
                "tensorflow": tf.__version__,
                "keras": keras.__version__,
                "devices": [d.name for d in tf.config.list_physical_devices()],
                "teacher_parameters": a.count_params(),
                "student_parameters": b.count_params(),
                "pi_zero_verified": False,
                "status": "local_forward_pass_ok",
            }
        elif args.command == "pretrained":
            from .pretrained import import_weights

            result = import_weights(args.output, args.size)
        elif args.command == "synthetic":
            from .data import synthetic

            result = {"manifest": synthetic(args.output, config, args.per_class)}
        elif args.command == "prepare-coco":
            from .data import prepare_coco

            result = {
                "manifest": prepare_coco(
                    args.source, args.output, config, args.sessions, args.trust_source_splits
                )
            }
        elif args.command == "validate":
            doc = validate(args.manifest, config)
            result = {
                "samples": len(doc["samples"]),
                "synthetic": doc["synthetic"],
                "split_provenance": doc.get("split_provenance"),
                "valid": True,
            }
        elif args.command == "train":
            from .training import run

            result = run(args.manifest, config, args.output, args.pretrained, args.smoke)
        elif args.command == "train-teacher":
            from .teacher_training import train_teacher

            report = train_teacher(
                args.manifest,
                config,
                args.output,
                args.pretrained,
                args.smoke,
                args.resume,
                args.epoch_budget,
                initial_teacher=args.initial_teacher,
            )
            result = {k: v for k, v in report.items() if k not in ("history", "validation")}
            result["output"] = str(Path(args.output).resolve())
            if "validation" in report:
                result["validation_metrics"] = report["validation"]["metrics"]
        elif args.command == "evaluate-teacher":
            from .teacher_training import evaluate_teacher

            report = evaluate_teacher(args.run, args.manifest, args.split, args.output)
            result = {
                "output": args.output,
                "split": args.split,
                "metrics": report["metrics"],
                "synthetic_training": report["synthetic_training"],
            }
        elif args.command == "predict-teacher":
            from .teacher_runtime import TeacherPredictor

            result = TeacherPredictor(args.model, args.allow_unvalidated).predict(args.image)
        elif args.command == "predict":
            from .runtime import Predictor

            result = Predictor(args.model, args.allow_unvalidated).predict(args.image)
        elif args.command == "benchmark":
            from .runtime import benchmark

            result = benchmark(args.model, args.image, args.count, args.allow_unvalidated)
        elif args.command == "replay":
            from .vision import replay

            result = replay(
                args.video,
                args.model,
                args.output,
                max_frames=args.max_frames,
                allow_unvalidated=args.allow_unvalidated,
            )
        print(json.dumps(result, indent=2))
    except Exception as exc:
        if args.command == "train":
            from .data import write_json

            # Don't overwrite an older run's status when the caller reused a directory.
            if not isinstance(exc, FileExistsError) and Path(args.output).is_dir():
                write_json(
                    Path(args.output) / "status.json", {"state": "failed", "error": str(exc)}
                )
        raise


if __name__ == "__main__":
    main()
