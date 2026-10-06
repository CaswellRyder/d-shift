"""End-to-end validation for real teachers or INT8 students with current proposals.

One-to-one, same-class IoU>=0.5 matching at the model's stored acceptance threshold.
No test data, retraining, camera access, or flight commands. Upstream incomplete
labels and correlated recordings limit interpretation of measured precision/recall.
"""

import argparse
import os
from collections import defaultdict
from pathlib import Path
import time
import shutil

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import cv2
import numpy as np
from PIL import Image
from dtr.data import read_json, sha256, write_json
from dtr.teacher_runtime import TeacherPredictor
from dtr.runtime import Predictor
from dtr.tracking import iou
from dtr.vision import DEFAULT_LIMIT, PROPOSAL_PROFILES, observe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile", choices=PROPOSAL_PROFILES, default="v2")
    parser.add_argument("--duplicate-policy", choices=["none", "nested"], default="none")
    parser.add_argument("--goal-limit", type=int, choices=[12, 24], default=12)
    parser.add_argument("--balloon-model", default="runs/balloon-dtr-v10-20261003/teacher.keras")
    parser.add_argument("--goal-model", default="runs/goal-dtr-v10-20261003/teacher.keras")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    snapshot = output.with_suffix(".vision.py")
    if snapshot.exists():
        raise FileExistsError(snapshot)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile("src/dtr/vision.py", snapshot)
    source = Path("data/roboflow-dtr-v10-grouped/coco/valid")
    annotations = source / "_annotations.coco.json"
    doc = read_json(annotations)
    names = {c["id"]: c["name"] for c in doc["categories"]}
    anns = defaultdict(list)
    for a in doc["annotations"]:
        anns[a["image_id"]].append(a)
    report = {
        "split": "val",
        "test_evaluated": False,
        "frames": len(doc["images"]),
        "image_size": [320, 240],
        "proposal_limit": DEFAULT_LIMIT,
        "proposal_limits": {"balloon": DEFAULT_LIMIT, "goal": args.goal_limit},
        "proposal_profile": args.profile,
        "duplicate_policy": args.duplicate_policy,
        "annotation_sha256": sha256(annotations),
        "vision_sha256": sha256(Path("src/dtr/vision.py")),
        "scope": "One-to-one same-class IoU>=0.5 at stored threshold; correlated validation only",
        "timing": "Local Mac processing, excludes camera/transport; includes first-call warmup",
        "tasks": {},
    }
    for task in ("balloon", "goal"):
        model = Path(getattr(args, f"{task}_model"))
        predictor = Predictor(model, True) if model.suffix == ".tflite" else TeacherPredictor(model, True)
        cfg = read_json(f"configs/{task}.json")
        stats = {
            k: dict(targets=0, true_positive=0, false_positive=0)
            for k in cfg["classes"]
            if k != "background"
        }
        durations, frames = [], []
        for idx, im in enumerate(doc["images"]):
            with Image.open(source / im["file_name"]) as opened:
                rgb = cv2.resize(np.asarray(opened.convert("RGB")), (320, 240))
            truth, ambiguous = [], []
            for a in anns[im["id"]]:
                x, y, w, h = a["bbox"]
                box = [x / 2, y * 240 / 640, (x + w) / 2, (y + h) * 240 / 640]
                name = names[a["category_id"]]
                label = cfg["aliases"].get(name)
                if label:
                    truth.append((label, box))
                    stats[label]["targets"] += 1
                elif name == "Balloons" and task == "balloon":
                    ambiguous.append(box)
            start = time.perf_counter()
            result = observe(
                rgb,
                predictor,
                limit=args.goal_limit if task == "goal" else DEFAULT_LIMIT,
                profile=args.profile,
                duplicate_policy=args.duplicate_policy,
            )
            durations.append((time.perf_counter() - start) * 1000)
            used, accepted = set(), []
            for pred in sorted(result["observations"], key=lambda p: -p["score"]):
                if not pred["accepted"]:
                    continue
                match = max(
                    (i for i, t in enumerate(truth) if i not in used and t[0] == pred["label"]),
                    key=lambda i: iou(pred["box"], truth[i][1]),
                    default=None,
                )
                good = match is not None and iou(pred["box"], truth[match][1]) >= 0.5
                if not good and any(iou(pred["box"], b) >= 0.5 for b in ambiguous):
                    continue
                if good:
                    used.add(match)
                stats[pred["label"]]["true_positive" if good else "false_positive"] += 1
                accepted.append(
                    {
                        "box": pred["box"],
                        "label": pred["label"],
                        "score": pred["score"],
                        "matched": good,
                    }
                )
            frames.append({"file": im["file_name"], "accepted": accepted})
            if idx % 100 == 0:
                print(task, idx, "of", len(doc["images"]), flush=True)
        for s in stats.values():
            s["precision"] = s["true_positive"] / max(1, s["true_positive"] + s["false_positive"])
            s["recall"] = s["true_positive"] / max(1, s["targets"])
        report["tasks"][task] = dict(
            model_sha256=sha256(model),
            threshold=predictor.metadata["threshold"],
            classes=stats,
            processing_ms_median=float(np.median(durations)),
            processing_ms_p95=float(np.percentile(durations, 95)),
            frames=frames,
        )
        write_json(output.with_suffix(".partial.json"), report)
    write_json(output, report)
    print(
        {task: {k: v for k, v in r.items() if k != "frames"} for task, r in report["tasks"].items()}
    )


if __name__ == "__main__":
    main()
