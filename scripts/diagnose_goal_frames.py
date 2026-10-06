"""Retain all goal scores and partition stage failures on train or development val only."""

import argparse
from collections import Counter, defaultdict
import os
from pathlib import Path
import shutil

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import cv2
import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, write_json
from dtr.diagnostics import diagnose_frame
from dtr.teacher_runtime import TeacherPredictor
from dtr.vision import observe, proposals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "valid"], required=True)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="runs/goal-proposals-20261005/teacher.keras")
    args = parser.parse_args()
    if args.stride < 1:
        parser.error("stride must be positive")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    for file in ("src/dtr/vision.py", "src/dtr/diagnostics.py", __file__):
        shutil.copyfile(file, output / Path(file).name)
    source = Path("data/roboflow-dtr-v10-grouped/coco") / args.split
    labels = source / "_annotations.coco.json"
    doc = read_json(labels)
    names = {c["id"]: c["name"] for c in doc["categories"]}
    aliases = read_json("configs/goal.json")["aliases"]
    annotations = defaultdict(list)
    for a in doc["annotations"]:
        annotations[a["image_id"]].append(a)
    predictor = TeacherPredictor(args.model, True)
    report = dict(
        split=args.split, stride=args.stride, image_size=[320, 240],
        model_sha256=sha256(args.model), annotation_sha256=sha256(labels),
        vision_sha256=sha256("src/dtr/vision.py"),
        diagnostics_sha256=sha256("src/dtr/diagnostics.py"),
        classes=predictor.metadata["classes"], threshold=predictor.metadata["threshold"],
        proposal_profile="balloon_components", proposal_limit=12, diagnostic_limit=64,
        duplicate_policy="nested", test_evaluated=False, training_approved=False,
        deployment_approved=False,
        scope="Annotation-relative triage, not corrected labels or independent accuracy. "
              "64-candidate geometry only; teacher sees at most 12 crops. "
              "Miss causes use furthest successful stage, not a causal intervention.",
        target_causes={}, prediction_causes={}, frames=[],
    )
    totals, predictions = defaultdict(Counter), defaultdict(Counter)
    images = sorted(doc["images"], key=lambda im: im["file_name"])[::args.stride]
    for index, im in enumerate(images):
        with Image.open(source / im["file_name"]) as opened:
            rgb = cv2.resize(np.asarray(opened.convert("RGB")), (320, 240))
        truth = []
        for a in annotations[im["id"]]:
            label = aliases.get(names[a["category_id"]])
            if label:
                x, y, w, h = a["bbox"]
                sx, sy = 320 / im["width"], 240 / im["height"]
                truth.append(dict(label=label, box=[x*sx, y*sy, (x+w)*sx, (y+h)*sy]))
        result = observe(rgb, predictor, limit=12, profile="balloon_components")
        diagnostic = proposals(rgb, "goal", limit=64, profile="balloon_components")
        audit = diagnose_frame(truth, result["observations"], diagnostic)
        for row in audit["targets"]:
            totals[row["label"]][row["cause"]] += 1
        for row in audit["predictions"]:
            predictions[result["observations"][row["index"]]["label"]][row["cause"]] += 1
        report["frames"].append(dict(
            file=im["file_name"], truth=truth, observations=result["observations"], **audit,
        ))
        if index % 100 == 0:
            print(index, "of", len(images), flush=True)
    report.update(target_causes=dict(totals), prediction_causes=dict(predictions))
    write_json(output / "report.json", report)
    print(dict(frames=len(images), target_causes=dict(totals), prediction_causes=dict(predictions)))


if __name__ == "__main__":
    main()
