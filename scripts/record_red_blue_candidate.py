"""Record actual training/export artifacts without opening the reserved test set."""
import argparse
import json
from pathlib import Path

from dtr.data import read_json, sha256
from dtr.runtime import Predictor
from dtr.vision import RED_BLUE_PROFILE, validate_model_profile


def record(teacher_run, student_run, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    teacher_run, student_run = Path(teacher_run), Path(student_run)
    provenance = read_json(student_run / "provenance.json")
    teacher_meta = read_json(teacher_run / "teacher.json")
    if sha256(teacher_run / "teacher.keras") != provenance["teacher_sha256"]:
        raise ValueError("Teacher checksum changed")
    validate_model_profile(teacher_meta, RED_BLUE_PROFILE)
    models = {}
    for encoding in ("int8", "float32"):
        path = student_run / f"student.{encoding}.tflite"
        predictor = Predictor(path, allow_unvalidated=True)
        validate_model_profile(predictor.metadata, RED_BLUE_PROFILE)
        models[encoding] = dict(path=str(path), sha256=sha256(path), bytes=path.stat().st_size,
                                metadata_sha256=sha256(path.with_suffix(".json")),
                                threshold=predictor.metadata["threshold"])
    report = dict(status="BOOTSTRAP_CANDIDATE_NOT_FLIGHT_READY",
                  teacher_sha256=teacher_meta["sha256"],
                  student_keras_sha256=sha256(student_run/"student.keras"),
                  manifest_sha256=provenance["manifest_sha256"],
                  config=provenance["config"], models=models,
                  teacher_validation=read_json(teacher_run/"validation.json"),
                  student_report=read_json(student_run/"report.json"),
                  test_evaluated=False, end_to_end_94_percent_established=False,
                  pi_zero_verified=False, deployment_approved=False)
    # Reports are committed evidence, not a copied model or raw dataset release.
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-run", required=True)
    parser.add_argument("--student-run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = record(args.teacher_run, args.student_run, args.output)
    print(json.dumps(dict(status=report["status"], models=report["models"]), indent=2))


if __name__ == "__main__":
    main()
