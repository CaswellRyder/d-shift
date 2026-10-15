"""Score the square-goal search on DTR v10 frames, one round color at a time.

Train reads the grouped COCO export (EngDes2 is never involved); val reads the
595-frame 320x240 set in runs/pi-comparison-20261005. The test split is never read.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from dtr.goal_square import GOAL_COLORS, SQUARE_LABELS, find_squares
from dtr.runtime import Predictor
from dtr.tracking import iou

SIZES = ((16, "<16"), (32, "16-32"), (64, "32-64"), (1e9, ">64"))
TRAIN = Path("data/roboflow-dtr-v10-grouped/coco/train")
VAL = Path("runs/pi-comparison-20261005")
COCO_SQUARES = {"Orange Square Goal": "orange_square", "Yellow Square Goal": "yellow_square"}


def bucket(box):
    side = max(box[2] - box[0], box[3] - box[1])
    return next(name for limit, name in SIZES if side < limit)


def frames(split, stride):
    if split == "val":
        for row in json.loads((VAL / "frames.json").read_text())["frames"][::stride]:
            bgr = cv2.imread(str(VAL / row["path"]))
            yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), row["truth"]
        return
    doc = json.loads((TRAIN / "_annotations.coco.json").read_text())
    names = {c["id"]: c["name"] for c in doc["categories"]}
    by_image = defaultdict(list)
    for ann in doc["annotations"]:
        by_image[ann["image_id"]].append(ann)
    for image in doc["images"][::stride]:
        sx, sy = 320 / image["width"], 240 / image["height"]
        bgr = cv2.resize(cv2.imread(str(TRAIN / image["file_name"])), (320, 240),
                         interpolation=cv2.INTER_AREA)
        truth = []
        for ann in by_image[image["id"]]:
            label = COCO_SQUARES.get(names[ann["category_id"]])
            if label:
                x, y, w, h = ann["bbox"]
                truth.append(dict(label=label, box=[x * sx, y * sy, (x + w) * sx, (y + h) * sy]))
        yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), truth


def evaluate(split, stride, predictor, color, max_crops, threshold):
    hits, total, false = Counter(), Counter(), Counter()
    calls, ms, count = 0, [], 0
    for rgb, truth in frames(split, stride):
        result = find_squares(rgb, predictor, (color,), threshold, max_crops)
        calls += result["neural_calls"]
        ms.append(result["search_ms"])
        count += 1
        used = set()
        for target in (t for t in truth if t["label"] == SQUARE_LABELS[color]):
            total[bucket(target["box"])] += 1
            match = next((i for i, d in enumerate(result["detections"])
                          if i not in used and iou(d["box"], target["box"]) >= 0.5), None)
            if match is not None:
                used.add(match)
                hits[bucket(target["box"])] += 1
        for i, detection in enumerate(result["detections"]):
            if i not in used:
                false[bucket(detection["box"])] += 1
    big = [name for _, name in SIZES[1:]]
    return dict(
        split=split, color=color, frames=count, max_crops=max_crops, threshold=threshold,
        recall_by_size={name: f"{hits[name]}/{total[name]}" for _, name in SIZES},
        recall_16px_up=round(sum(hits[b] for b in big) / max(1, sum(total[b] for b in big)), 3),
        false_positives_by_size=dict(false),
        false_positives_per_100_frames=round(100 * sum(false.values()) / count, 2),
        crops_per_frame=round(calls / count, 2),
        search_ms_mean=round(float(np.mean(ms)), 2),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--model", default="models/v1.0.0/goal/student.float.tflite")
    parser.add_argument("--max-crops", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.8)
    args = parser.parse_args()
    predictor = Predictor(args.model, allow_unvalidated=True)
    for color in GOAL_COLORS:
        print(json.dumps(evaluate(args.split, args.stride, predictor, color,
                                  args.max_crops, args.threshold)))


if __name__ == "__main__":
    main()
