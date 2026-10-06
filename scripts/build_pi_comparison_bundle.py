"""Freeze a self-contained research comparison bundle; no reserved test images."""
import argparse
from collections import Counter
from pathlib import Path
import shutil

import cv2
import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, write_json
from dtr.runtime import Predictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    for name in ("LICENSE", "THIRD_PARTY_NOTICES.md"):
        shutil.copyfile(name, out / name)
    (out / "dtr").mkdir()
    for name in ("__init__", "data", "runtime", "native_tflite", "goal_evidence", "vision", "tracking"):
        shutil.copyfile(f"src/dtr/{name}.py", out / "dtr" / f"{name}.py")
    for name in ("pi_compare", "pi_live_compare"):
        shutil.copyfile(f"scripts/{name}.py", out / f"{name}.py")
    shutil.copyfile(args.baseline, out / "baseline.py")
    (out / "models").mkdir()
    (out / "golden").mkdir()
    golden = []
    for task in ("goal", "balloon"):
        run = Path(f"runs/{task}-pi-student-20261005")
        for suffix in ("tflite", "json"):
            shutil.copyfile(run / f"student.int8.{suffix}", out / "models" / f"{task}.{suffix}")
        model = Predictor(out / "models" / f"{task}.tflite", allow_unvalidated=True)
        manifest = Path(f"data/dtr-proposals-reviewed-20261005/{task}/manifest.json")
        counts = Counter()
        for row in read_json(manifest)["samples"]:
            if row["split"] != "val" or counts[row["label"]] >= 3:
                continue
            counts[row["label"]] += 1
            path = f"golden/{task}-{len(golden):03}.png"
            with Image.open(manifest.parent / row["path"]) as opened:
                opened.convert("RGB").save(out / path)
            prediction = model.predict(out / path)
            golden.append(dict(task=task, path=path, sha256=sha256(out / path),
                               label=prediction["label"], scores=prediction["scores"]))
    write_json(out / "golden.json", golden)
    source = Path("data/roboflow-dtr-v10-grouped/coco/valid")
    coco = read_json(source / "_annotations.coco.json")
    names = {c["id"]: c["name"] for c in coco["categories"]}
    cfg = read_json("configs/goal.json")
    labels = {key: cfg["aliases"].get(value) for key, value in names.items()}
    by_image = {}
    for ann in coco["annotations"]:
        if labels.get(ann["category_id"]):
            by_image.setdefault(ann["image_id"], []).append(ann)
    (out / "frames").mkdir()
    frames = []
    for index, info in enumerate(coco["images"]):
        path = f"frames/{index:04}.png"
        with Image.open(source / info["file_name"]) as opened:
            original = np.array(opened.convert("RGB"))
        height, width = original.shape[:2]
        Image.fromarray(cv2.resize(original, (320, 240))).save(out / path)
        truth = []
        for ann in by_image.get(info["id"], []):
            x, y, w, h = ann["bbox"]
            truth.append(dict(label=labels[ann["category_id"]],
                              box=[x*320/width, y*240/height, (x+w)*320/width, (y+h)*240/height]))
        frames.append(dict(path=path, sha256=sha256(out / path), source=info["file_name"], truth=truth))
    write_json(out / "frames.json", dict(split="development_validation", size=[320, 240],
               annotation_sha256=sha256(source / "_annotations.coco.json"), frames=frames,
               caveat="Reused development data; correlated scenes and upstream-label uncertainty; not held-out final accuracy"))
    files = {str(p.relative_to(out)): sha256(p) for p in out.rglob("*") if p.is_file()}
    write_json(out / "bundle.json", dict(files=files, baseline_sha256=sha256(args.baseline),
               deployment_approved=False, purpose="two-method research comparison"))
    print({"bundle": str(out), "frames": len(frames), "golden_crops": len(golden)})


if __name__ == "__main__":
    main()
