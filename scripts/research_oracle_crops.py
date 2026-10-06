"""Separate source detail, crop identity and proposal coverage on development data.

Oracle boxes are supplied by labels, NOT found by the system. Positive-only crop
accuracy is not detection precision/recall. No training or reserved-test access.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, write_json
from dtr.runtime import Predictor
from dtr.tracking import iou
from dtr.vision import proposals


def size_bin(box):
    side = max(box[2] - box[0], box[3] - box[1])
    return "lt8" if side < 8 else "8to16" if side < 16 else "16to32" if side < 32 else "ge32"


def oracle_crop(rgb, box, padding=.12):
    """Continuous label box -> containing integer crop with explicit context."""
    height, width = rgb.shape[:2]
    left, top, right, bottom = map(float, box)
    if not np.isfinite([left, top, right, bottom, padding]).all() or padding < 0:
        raise ValueError("Invalid oracle coordinates")
    if right <= left or bottom <= top:
        raise ValueError("Empty oracle box")
    pad = max(right-left, bottom-top) * padding
    x1, y1 = max(0, int(np.floor(left-pad))), max(0, int(np.floor(top-pad)))
    x2, y2 = min(width, int(np.ceil(right+pad))), min(height, int(np.ceil(bottom+pad)))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Oracle crop outside image")
    return rgb[y1:y2, x1:x2]


def summarize(rows):
    counts = defaultdict(lambda: dict(n=0, correct=0, accepted_correct=0, background=0,
                                     correct_color=0, covered12=0, covered64=0))
    for row in rows:
        for category in ("all", row["truth"], "size:" + row["size_bin"]):
            cell = counts[(row["model"], row["resolution"], category)]
            got = row["prediction"]
            cell["n"] += 1
            cell["correct"] += got["label"] == row["truth"]
            cell["accepted_correct"] += got["label"] == row["truth"] and got["accepted"]
            cell["background"] += got["label"] == "background"
            cell["correct_color"] += got["label"].split("_")[0] == row["truth"].split("_")[0]
            for key in ("covered12", "covered64"):
                cell[key] += row[key]
    return [dict(model=model, resolution=resolution, category=category, **cell,
                 accuracy=cell["correct"]/cell["n"],
                 accepted_correct_rate=cell["accepted_correct"]/cell["n"])
            for (model, resolution, category), cell in sorted(counts.items())]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="runs/pi-comparison-20261005")
    parser.add_argument("--source", default="data/roboflow-dtr-v10-grouped/coco/valid")
    parser.add_argument("--model", action="append", required=True, help="name=model.keras or .tflite")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() or output.with_suffix(".rows.jsonl").exists():
        raise FileExistsError(output)
    base, source = Path(args.base), Path(args.source).resolve()
    manifest = read_json(base / "frames.json")
    if (manifest.get("split") != "development_validation" or manifest.get("size") != [320,240]
            or sha256(source / "_annotations.coco.json") != manifest["annotation_sha256"]):
        raise ValueError("Wrong development frames/annotations")
    cv2.setNumThreads(1)
    models, identities = {}, {}
    for spec in args.model:
        name, path = spec.split("=", 1)
        if name in models:
            raise ValueError("Duplicate model name")
        if Path(path).suffix == ".keras":
            from dtr.teacher_runtime import TeacherPredictor
            model = TeacherPredictor(path, True)
        else:
            model = Predictor(path, True)
        if model.metadata["task"] != "goal":
            raise ValueError("Expected goal model")
        models[name] = model
        identities[name] = dict(path=str(Path(path).resolve()), sha256=sha256(path))
    output.parent.mkdir(parents=True, exist_ok=True)
    all_rows, source_hashes = [], {}
    with output.with_suffix(".rows.jsonl").open("x") as stream:
        for index, frame in enumerate(manifest["frames"]):
            original_path = (source / frame["source"]).resolve()
            low_path = (base / frame["path"]).resolve()
            if not original_path.is_relative_to(source) or not low_path.is_relative_to(base.resolve()):
                raise ValueError("Image escapes source directory")
            if sha256(low_path) != frame["sha256"]:
                raise ValueError("Frozen frame changed")
            source_hashes[frame["source"]] = sha256(original_path)
            with Image.open(original_path) as im:
                original = np.array(im.convert("RGB"))
            with Image.open(low_path) as im:
                low = np.array(im.convert("RGB"))
            high = cv2.resize(original, (640, 480))
            for resolution, rgb, scale in (("320x240", low, 1), ("640x480", high, 2)):
                candidates = proposals(rgb, "goal", limit=64, profile="balloon_components")
                for target in frame["truth"]:
                    box = [v*scale for v in target["box"]]
                    crop = oracle_crop(rgb, box)
                    coverage = [iou(p["box"], box) >= .5 for p in candidates]
                    for name, model in models.items():
                        prediction = model.predict(crop)
                        row = dict(frame=frame["path"], truth=target["label"], box=box,
                                   size_bin=size_bin(target["box"]), crop_size=list(crop.shape[:2]),
                                   model=name, resolution=resolution, prediction=prediction,
                                   covered12=any(coverage[:12]), covered64=any(coverage))
                        stream.write(json.dumps(row) + "\n")
                        all_rows.append(row)
            if index % 100 == 0:
                print(index+1, "of", len(manifest["frames"]), flush=True)
    write_json(output, dict(
        models=identities, frames_sha256=sha256(base / "frames.json"),
        source_image_sha256=source_hashes, annotation_sha256=manifest["annotation_sha256"],
        script_sha256=sha256(__file__), vision_sha256=sha256(Path(__file__).resolve().parents[1] / "src/dtr/vision.py"),
        results=summarize(all_rows), test_evaluated=False, deployment_approved=False,
        scope="Oracle positive crops only; no negatives, no estimated range, no detection accuracy",
        padding=.12, size_bins="Longest truth side in 320x240 frame; not physical distance",
        proposal_caveat="Identical fixed pixel thresholds/kernels at both resolutions; not tuned high-resolution proposals"))
    print([r for r in summarize(all_rows) if r["category"] == "all"], flush=True)


if __name__ == "__main__":
    main()
