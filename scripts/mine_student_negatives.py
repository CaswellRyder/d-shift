"""Mine a bounded, quarantined review queue from approved TRAIN sources only.

No validation/test pixels. Zero overlap is only a candidate rule, not label proof:
every admitted crop must subsequently be visually reviewed. Original data unchanged.
"""
import argparse
from collections import defaultdict
import hashlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from dtr.data import read_json, sha256, write_json, overlap
from dtr.runtime import Predictor
from dtr.vision import proposals


def safe_background(candidate, truth):
    """Guard all annotated goals, including near misses and padded crop context."""
    a, b, c, d = candidate["crop_box"]
    pad = 3
    return not any(overlap([a-pad, b-pad, c+pad, d+pad], box) > 0 for box in truth)


def training_sources(doc):
    held = {r.get("source_image") for r in doc["samples"] if r["split"] != "train"}
    sources = {}
    for row in doc["samples"]:
        name = row.get("source_image", "")
        if row["split"] == "train":
            if not name.startswith("train/") or name in held or ".." in Path(name).parts:
                raise ValueError("Training source crosses split/path boundary")
            sources[name] = row["session"]
    return sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", default="data/roboflow-dtr-v10-grouped/coco")
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=96)
    args = parser.parse_args()
    if not 1 <= args.count <= 128:
        parser.error("Review queue must be bounded 1..128")
    out, source = Path(args.output), Path(args.source).resolve()
    if out.exists():
        raise FileExistsError(out)
    doc = read_json(args.manifest)
    if doc.get("synthetic", True) or doc.get("training_approved") is not True or doc["task"] != "goal":
        raise ValueError("Require approved real goal manifest")
    sources = training_sources(doc)
    cfg = read_json("configs/goal-proposals.json")
    annotations_path = source / "train/_annotations.coco.json"
    coco = read_json(annotations_path)
    names = {c["id"]: c["name"] for c in coco["categories"]}
    annotations = defaultdict(list)
    for row in coco["annotations"]:
        annotations[row["image_id"]].append(row)
    predictor = Predictor(args.model, True)
    if predictor.metadata["task"] != "goal" or predictor.metadata["classes"] != doc["classes"]:
        raise ValueError("Model taxonomy mismatch")
    out.mkdir(parents=True)
    cv2.setNumThreads(1)
    candidates, seen = [], set()
    images = sorted(coco["images"], key=lambda im: im["file_name"])
    processed = 0
    for info in images:
        key = "train/" + info["file_name"]
        if key not in sources:
            continue
        path = (source / key).resolve()
        if not path.is_relative_to(source / "train"):
            raise ValueError("Source path escape")
        with Image.open(path) as im:
            rgb = cv2.resize(np.array(im.convert("RGB")), (320, 240))
        truth = []
        for ann in annotations[info["id"]]:
            if names[ann["category_id"]] in cfg["aliases"]:
                x, y, w, h = ann["bbox"]
                truth.append([x*320/info["width"], y*240/info["height"],
                              (x+w)*320/info["width"], (y+h)*240/info["height"]])
        found = []
        for candidate in proposals(rgb, "goal", limit=12, profile="balloon_components"):
            if not safe_background(candidate, truth):
                continue
            a, b, c, d = candidate["crop_box"]
            crop = rgb[b:d, a:c]
            got = predictor.predict(crop)
            if got["accepted"]:
                digest = hashlib.sha256(str(crop.shape).encode()+crop.tobytes()).hexdigest()
                if digest not in seen:
                    found.append(dict(source_image=key, source_file_sha256=sha256(path),
                                      session=sources[key], split="train", label="background",
                                      box=candidate["box"], crop_box=candidate["crop_box"],
                                      prediction=got["label"], score=got["score"], digest=digest))
        # At most one crop per source; prioritize the known orange-square weakness.
        found.sort(key=lambda r: (r["prediction"] == "orange_square", r["score"]), reverse=True)
        if found:
            candidates.append(found[0])
            seen.add(found[0]["digest"])
        processed += 1
        if processed % 500 == 0:
            print("Training frames", processed, "candidates", len(candidates), flush=True)
    candidates.sort(key=lambda r: (r["prediction"] == "orange_square", r["score"]), reverse=True)
    selected = candidates[:args.count]
    (out / "crops").mkdir()
    for index, row in enumerate(selected):
        with Image.open(source / row["source_image"]) as im:
            rgb = cv2.resize(np.array(im.convert("RGB")), (320, 240))
        a, b, c, d = row["crop_box"]
        relative = f"crops/{index:03d}.png"
        Image.fromarray(rgb[b:d, a:c]).save(out / relative)
        row.update(path=relative, sha256=sha256(out / relative), review_index=index)
    write_json(out / "queue.json", selected)
    write_json(out / "receipt.json", dict(
        base_manifest_sha256=sha256(args.manifest), model_sha256=sha256(args.model),
        annotations_sha256=sha256(annotations_path), queue_sha256=sha256(out / "queue.json"),
        script_sha256=sha256(__file__), vision_sha256=sha256("src/dtr/vision.py"),
        frames_considered=processed, candidates=len(candidates), selected=len(selected),
        training_approved=False, test_evaluated=False, label_review="required; zero overlap is not sufficient",
        synthetic=False, source=str(source)))
    for page in range((len(selected)+23)//24):
        sheet = Image.new("RGB", (6*190, 4*180), "white")
        draw = ImageDraw.Draw(sheet)
        for n, row in enumerate(selected[page*24:(page+1)*24]):
            with Image.open(out / row["path"]) as im:
                tile = ImageOps.contain(im, (184, 145))
            x, y = n%6*190, n//6*180
            sheet.paste(tile, (x, y))
            draw.text((x, y+147), f'{row["review_index"]}: {row["prediction"]}', fill="black")
            draw.text((x, y+162), f'{row["score"]:.4f}', fill="black")
        sheet.save(out / f"review-{page}.png")
    print(dict(frames=processed, candidates=len(candidates), queued=len(selected), output=str(out)), flush=True)


if __name__ == "__main__":
    main()
