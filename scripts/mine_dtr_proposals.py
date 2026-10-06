"""Train-only proposal adaptation; upstream labels, not teacher pseudo-labels.

Background requires zero overlap of the padded crop with ANY task target (including
generic balloons). Teacher scores rank only how confusing safe negative candidates
are. Unknown/incomplete upstream labels remain a risk: visually review output sheets.
Existing train/val/test base samples remain unchanged; no new val/test examples.
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import argparse
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from dtr.data import read_json, write_json, sha256, validate, overlap
from dtr.teacher_runtime import TeacherPredictor
from dtr.tracking import iou
from dtr.vision import proposals

BASE = Path("data/roboflow-dtr-v10-grouped")


def assign(candidate, truth):
    """Ignore ambiguous overlap; match positives at IoU .5, never auto-color Balloons."""
    if truth:
        label, box = max(truth, key=lambda t: iou(candidate["box"], t[1]))
        if iou(candidate["box"], box) >= 0.5:
            return label  # None for generic balloons: exclude rather than guess.
        if any(overlap(candidate["crop_box"], box) > 0 for _, box in truth):
            return None
    return "background"


def build(task, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    cfg = read_json(f"configs/{task}.json")
    manifest = BASE / task / "manifest.json"
    original = read_json(manifest)
    source = BASE / "coco/train"
    coco = read_json(source / "_annotations.coco.json")
    names = {c["id"]: c["name"] for c in coco["categories"]}
    annotations = defaultdict(list)
    for a in coco["annotations"]:
        annotations[a["image_id"]].append(a)
    groups = read_json(BASE / "groups.json")
    model = Path(f"runs/{task}-dtr-v10-20261003/teacher.keras")
    predictor = TeacherPredictor(model, True)
    output.mkdir(parents=True)
    rows = []
    for row in original["samples"]:
        dest = output / row["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.hardlink_to(manifest.parent / row["path"])
        rows.append(dict(row))
    seen = {r["sha256"] for r in rows}
    mined, exclusions = [], Counter()
    rng = np.random.default_rng(42)
    # Fixed deterministic sample of training frames, not validation-error-guided selection.
    images = sorted(coco["images"], key=lambda im: im["file_name"])[::2]
    for idx, info in enumerate(images):
        key = "train/" + info["file_name"]
        with Image.open(source / info["file_name"]) as opened:
            full = opened.convert("RGB")
            source_hash = hashlib.sha256(full.tobytes()).hexdigest()
            rgb = cv2.resize(np.asarray(full), (320, 240))
        truth = []
        for ann in annotations[info["id"]]:
            name = names[ann["category_id"]]
            label = cfg["aliases"].get(name)
            if label or (task == "balloon" and name == "Balloons"):
                x, y, w, h = ann["bbox"]
                truth.append(
                    (
                        label,
                        [
                            x * 320 / info["width"],
                            y * 240 / info["height"],
                            (x + w) * 320 / info["width"],
                            (y + h) * 240 / info["height"],
                        ],
                    )
                )
        candidates = proposals(rgb, task)
        crops = [
            rgb[c["crop_box"][1] : c["crop_box"][3], c["crop_box"][0] : c["crop_box"][2]]
            for c in candidates
        ]
        predictions = predictor.predict_many(crops)
        negatives, positives = [], []
        for c, crop, pred in zip(candidates, crops, predictions):
            label = assign(c, truth)
            if label is None:
                exclusions["ambiguous_overlap"] += 1
                continue
            record = (label, c, crop, pred)
            if label == "background":
                if pred["accepted"]:
                    negatives.append(record)
                else:
                    exclusions["already_rejected_background"] += 1
            else:
                positives.append(record)
        negatives = sorted(negatives, key=lambda r: -r[3]["score"])[:2]
        rng.shuffle(positives)
        for label, c, crop, pred in negatives + positives[:3]:
            digest = hashlib.sha256(str(crop.shape).encode() + crop.tobytes()).hexdigest()
            rel = f"mined/{digest}.png"
            dest = output / rel
            dest.parent.mkdir(exist_ok=True)
            Image.fromarray(crop).save(dest)
            image_hash = sha256(dest)
            if image_hash in seen:
                exclusions["duplicate_crop"] += 1
                continue
            seen.add(image_hash)
            row = dict(
                path=rel,
                label=label,
                split="train",
                session=groups[key]["session"],
                sha256=image_hash,
                source_image=key,
                source_sha256=source_hash,
                box=c["box"],
                crop_box=c["crop_box"],
                origin="train_only_proposal_overlap_rule",
                previous_prediction=pred["label"],
                previous_score=pred["score"],
            )
            rows.append(row)
            mined.append(row)
        if idx % 200 == 0:
            print(task, idx, "of", len(images), "added", len(mined), flush=True)
    receipt = dict(
        task=task,
        base_manifest_sha256=sha256(manifest),
        source_annotation_sha256=sha256(source / "_annotations.coco.json"),
        proposal_source_sha256=sha256("src/dtr/vision.py"),
        teacher_sha256=sha256(model),
        training_frames_considered=len(images),
        added=dict(Counter(r["label"] for r in mined)),
        exclusions=dict(exclusions),
        new_samples_train_only=True,
        val_and_test_samples_unchanged=True,
        labeling="Upstream IoU >= .5 positives; zero padded-crop overlap negatives; ambiguity excluded",
        review="Needs representative visual review; not a full human annotation audit",
    )
    write_json(output / "mining-receipt.json", receipt)
    qualification = {
        **original["qualification"],
        "proposal_adaptation": receipt["labeling"],
        "mining_review": receipt["review"],
        "test_evaluated": False,
    }
    write_json(
        output / "manifest.json",
        {
            **original,
            "training_approved": False,
            "qualification": qualification,
            "samples": rows,
            "mining_receipt_sha256": sha256(output / "mining-receipt.json"),
        },
    )
    validate(output / "manifest.json", cfg)
    write_json(output / "mined.json", mined)
    for label in cfg["classes"]:
        examples = [r for r in mined if r["label"] == label]
        count = min(64 if label == "background" else 16, len(examples))
        if not count:
            continue
        chosen = rng.choice(len(examples), count, replace=False)
        sheet = Image.new("RGB", (8 * 140, ((count + 7) // 8) * 165), "white")
        draw = ImageDraw.Draw(sheet)
        for index, n in enumerate(chosen):
            row = examples[int(n)]
            with Image.open(output / row["path"]) as im:
                tile = ImageOps.contain(im, (135, 135))
            x, y = index % 8 * 140, index // 8 * 165
            sheet.paste(tile, (x, y))
            draw.text((x, y + 136), f"{index}: {label}", fill="black")
            draw.text((x, y + 149), row["previous_prediction"], fill="black")
        sheet.save(output / f"review-{label}.jpg")
        write_json(output / f"review-{label}.json", [examples[int(n)] for n in chosen])
    print(receipt, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["balloon", "goal"], required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    build(args.task, args.output)
