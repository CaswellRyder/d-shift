"""Offboard cached YOLO boxes + existing MobileNetV4 verifier experiment; no training."""

import argparse
import math
import os
from pathlib import Path

from dtr.data import read_json, sha256, write_json
from dtr.detector_metrics import summarize
from dtr.tracking import iou


def crop_box(box, width, height):
    padding = .12 * max(box[2]-box[0], box[3]-box[1])
    return [max(0, math.floor(box[0]-padding)), max(0, math.floor(box[1]-padding)),
            min(width, math.ceil(box[2]+padding)), min(height, math.ceil(box[3]+padding))]


def verified_predictions(predictions, verifications):
    if len(predictions) != len(verifications):
        raise ValueError("Each candidate must have exactly one verifier result")
    candidates = []
    for prediction, verification in zip(predictions, verifications):
        if verification["accepted"]:
            candidates.append(dict(prediction, label=verification["label"],
                                   detector_label=prediction["label"],
                                   verifier_score=verification["score"]))
    # Class identities can change after verification. Apply the detector's same-class NMS
    # again so formerly different-class duplicates are not counted as independent objects.
    kept = []
    for prediction in sorted(candidates, key=lambda p: -p["score"]):
        if not any(prediction["label"] == p["label"] and iou(prediction["box"], p["box"]) > .7
                   for p in kept):
            kept.append(prediction)
    return kept


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, help="Immutable interim detector report")
    parser.add_argument("--verifier", default="runs/goal-proposals-20261005/teacher.keras")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    report_path = Path(args.report)
    report = read_json(report_path)
    if report["split"] != "val" or not report["interim"] or report["metrics"]["test_evaluated"]:
        raise ValueError("Only immutable interim development reports are supported")
    summarize(report["frames"], report["standard"])
    snapshot = read_json(report_path.parent / "snapshot.json")
    if snapshot["model_sha256"] != report["model_sha256"]:
        raise ValueError("Snapshot and detector report differ")
    dataset = Path(snapshot["dataset_root"])
    if sha256(dataset / "receipt.json") != snapshot["experiment"]["dataset_receipt_sha256"]:
        raise ValueError("Dataset receipt changed")
    receipt = read_json(dataset / "receipt.json")["splits"]["valid"]
    source = Path("data/roboflow-dtr-v10-grouped/coco/valid")
    if sha256(source / "_annotations.coco.json") != report["annotation_sha256"]:
        raise ValueError("Validation annotations changed")
    expected = {row["file"]: row["sha256"] for row in receipt["images"]}
    if set(expected) != {frame["file"] for frame in report["frames"]}:
        raise ValueError("Validation frame membership changed")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")  # Do not compete with the running MPS trainer.
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import numpy as np
    from PIL import Image
    from dtr.teacher_runtime import TeacherPredictor
    verifier = TeacherPredictor(args.verifier, allow_unvalidated=True)
    if verifier.metadata["task"] != "goal" or verifier.metadata["classes"] != [
            "background", *report["standard"]["classes"]]:
        raise ValueError("Verifier has wrong goal taxonomy")
    frames, decisions = [], []
    for index, frame in enumerate(report["frames"]):
        path = source / frame["file"]
        if sha256(path) != expected[frame["file"]]:
            raise ValueError("Validation image changed")
        with Image.open(path) as opened:
            rgb = np.asarray(opened.convert("RGB"))
        if list(rgb.shape[1::-1]) != frame["image_size"]:
            raise ValueError("Image dimensions differ from cached detector coordinates")
        predictions = frame["predictions"]
        boxes = [crop_box(p["box"], rgb.shape[1], rgb.shape[0]) for p in predictions]
        crops = [rgb[y1:y2, x1:x2] for x1, y1, x2, y2 in boxes]
        results = []
        for start in range(0, len(crops), 64):
            results.extend(verifier.predict_many(crops[start:start+64]))
        frames.append(dict(frame, predictions=verified_predictions(predictions, results)))
        decisions.append(dict(file=frame["file"], candidates=[
            dict(detector=p, crop_box=b, verifier=v) for p, b, v in zip(predictions, boxes, results)]))
        if index % 100 == 0:
            print(index, "of", len(report["frames"]), flush=True)
    if sha256(args.verifier) != verifier.metadata["sha256"]:
        raise ValueError("Verifier checkpoint changed")
    result = dict(report, frames=frames, metrics=summarize(frames, report["standard"]),
                  parent_report_sha256=sha256(report_path), candidate_decisions=decisions,
                  experiment="cached_detector_with_mobilenet_verifier", selected=False,
                  verifier_sha256=verifier.metadata["sha256"],
                  verifier_metadata_sha256=sha256(Path(args.verifier).with_suffix(".json")),
                  verifier_threshold=verifier.metadata["threshold"], crop_padding_fraction=.12,
                  score_contract="Raw detector score retained; verifier acceptance is a separate gate",
                  verifier_device="cpu", post_verifier_same_class_nms_iou=.7,
                  script_sha256=sha256(__file__),
                  scope="Offboard two-model development ablation, not deployment or Pi performance")
    write_json(output, result)
    print(result["metrics"]["classes"])


if __name__ == "__main__":
    main()
