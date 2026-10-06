"""Validation-only, class-agnostic proposal coverage; NOT detector precision/recall.

Checks whether the current viewer's bounded OpenCV candidates overlap each target
box. No neural inference, training, camera access, test evaluation, or flight commands.
"""

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, write_json
from dtr.vision import DEFAULT_LIMIT, proposals


def iou(a, b):
    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - overlap
    return overlap / union if union > 0 else 0.0


def evaluate(source, output, limit=DEFAULT_LIMIT):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(output)
    labels = source / "valid/_annotations.coco.json"
    doc = read_json(labels)
    categories = {r["id"]: r["name"] for r in doc["categories"]}
    annotations = defaultdict(list)
    for ann in doc["annotations"]:
        annotations[ann["image_id"]].append(ann)
    coverage = defaultdict(list)
    candidate_counts = defaultdict(list)
    for info in doc["images"]:
        with Image.open(source / "valid" / info["file_name"]) as opened:
            rgb = cv2.resize(np.asarray(opened.convert("RGB")), (320, 240))
        candidates = {task: proposals(rgb, task, limit=limit) for task in ("balloon", "goal")}
        for task, found in candidates.items():
            candidate_counts[task].append(len(found))
        for ann in annotations[info["id"]]:
            name = categories[ann["category_id"]]
            if name == "Balloons":
                continue
            task = "balloon" if name.endswith("Balloon") else "goal"
            x, y, w, h = ann["bbox"]
            box = [
                x * 320 / info["width"],
                y * 240 / info["height"],
                (x + w) * 320 / info["width"],
                (y + h) * 240 / info["height"],
            ]
            coverage[name].append(max((iou(box, c["box"]) for c in candidates[task]), default=0))
    report = {
        "split": "val",
        "frames": len(doc["images"]),
        "annotation_sha256": sha256(labels),
        "image_size": [320, 240],
        "proposal_limit_per_task": limit,
        "vision_source_sha256": sha256(Path(__file__).parents[1] / "src/dtr/vision.py"),
        "semantics": "Any proposal overlaps target; class agnostic. Not end-to-end detection metrics.",
        "scope": "All non-generic labeled boxes, including small targets omitted from crop training",
        "preprocessing_caveat": "OpenCV resize emulates viewer dimensions; browser JPEG processing may differ",
        "model_inference": False,
        "test_evaluated": False,
        "mean_candidates_per_frame": {k: float(np.mean(v)) for k, v in candidate_counts.items()},
        "classes": {
            k: {
                "targets": len(v),
                "coverage_iou_0.3": float(np.mean(np.array(v) >= 0.3)),
                "coverage_iou_0.5": float(np.mean(np.array(v) >= 0.5)),
            }
            for k, v in sorted(coverage.items())
        },
    }
    write_json(output, report)
    print(output.read_text())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Curated COCO root")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    args = parser.parse_args()
    evaluate(args.source, args.output, args.limit)
