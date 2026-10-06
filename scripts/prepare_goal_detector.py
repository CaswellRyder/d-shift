"""Convert grouped DTR train/valid goal annotations; never read or export reserved test."""

import argparse
from collections import Counter, defaultdict
import math
from pathlib import Path

import yaml

from dtr.data import read_json, sha256, write_json
from dtr.tracking import iou


def reviewed_training_drops(source, config, review_path):
    """Bind visual decisions to immutable training images and exact annotation IDs."""
    if review_path is None:
        return set(), None
    review = read_json(review_path)
    annotations = Path(source) / "train/_annotations.coco.json"
    if review["scope"] != "train_only" or sha256(annotations) != review["annotation_sha256"]:
        raise ValueError("Training review scope or annotation checksum mismatch")
    doc = read_json(annotations)
    images = {im["id"]: im for im in doc["images"]}
    anns = {ann["id"]: ann for ann in doc["annotations"]}
    categories = {c["id"]: config["aliases"].get(c["name"]) for c in doc["categories"]}
    drops, retains = set(), set()
    for decision in review["decisions"]:
        name = decision["file"]
        if Path(name).name != name or name.lower().startswith("highbay"):
            raise ValueError("Invalid training review image")
        if sha256(Path(source) / "train" / name) != decision["image_sha256"]:
            raise ValueError("Reviewed training image changed")
        drop, retain = anns[decision["drop_annotation_id"]], anns[decision["retain_annotation_id"]]
        if (drop["id"] == retain["id"] or drop["id"] in drops
                or drop["image_id"] != retain["image_id"]
                or images[drop["image_id"]]["file_name"] != name
                or categories[retain["category_id"]] != decision["retain_label"]
                or not categories[drop["category_id"]]):
            raise ValueError("Invalid reviewed annotation pair")
        boxes = [[a["bbox"][0], a["bbox"][1], a["bbox"][0]+a["bbox"][2],
                  a["bbox"][1]+a["bbox"][3]] for a in (drop, retain)]
        if iou(*boxes) < .7:
            raise ValueError("Reviewed annotations are not overlapping duplicates")
        drops.add(drop["id"])
        retains.add(retain["id"])
    if drops & retains:
        raise ValueError("Review drops a retained annotation")
    return drops, dict(path=str(Path(review_path).resolve()), sha256=sha256(review_path),
                       dropped_annotation_ids=sorted(drops), scope="train_only",
                       reviewer=review["reviewer"])


def yolo_box(box, width, height):
    if width <= 0 or height <= 0 or len(box) != 4 or not all(math.isfinite(v) for v in box):
        raise ValueError("Invalid image dimensions or box")
    x, y, w, h = box
    if w <= 0 or h <= 0:
        raise ValueError("Non-positive annotation extent")
    x1, y1, x2, y2 = max(0, x), max(0, y), min(width, x+w), min(height, y+h)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Annotation lies outside image")
    return [(x1+x2)/(2*width), (y1+y2)/(2*height), (x2-x1)/width, (y2-y1)/height]


def reviewed_training_corrections(source, config, path):
    """Explicit visual corrections only; never infer labels from color scores."""
    if path is None:
        return {}, None
    review = read_json(path)
    annotations = Path(source) / "train/_annotations.coco.json"
    if review["scope"] != "train_only" or sha256(annotations) != review["annotation_sha256"]:
        raise ValueError("Training correction scope or annotation checksum mismatch")
    doc = read_json(annotations)
    images = {im["id"]: im for im in doc["images"]}
    anns = {ann["id"]: ann for ann in doc["annotations"]}
    categories = {c["id"]: config["aliases"].get(c["name"]) for c in doc["categories"]}
    labels = set(config["classes"]) - {"background"}
    changes = {}
    for decision in review["decisions"]:
        name, ident = decision["file"], decision["annotation_id"]
        if Path(name).name != name or name.lower().startswith("highbay"):
            raise ValueError("Invalid training correction image")
        if ident not in anns or ident in changes:
            raise ValueError("Unknown or repeated correction annotation")
        ann = anns[ident]
        if (images[ann["image_id"]]["file_name"] != name
                or categories[ann["category_id"]] not in labels
                or categories[ann["category_id"]] != decision["original_label"]
                or ann["bbox"] != decision["original_bbox"]
                or sha256(Path(source) / "train" / name) != decision["image_sha256"]
                or not decision["reason"].strip()):
            raise ValueError("Training correction evidence mismatch")
        action, replacement = decision["action"], decision["replacement_label"]
        if action == "relabel" and replacement in labels and replacement != decision["original_label"]:
            changes[ident] = replacement
        elif action == "remove_non_goal" and replacement is None:
            changes[ident] = None
        else:
            raise ValueError("Invalid training correction action or replacement")
    return changes, dict(path=str(Path(path).resolve()), sha256=sha256(path),
                         annotation_changes=changes, scope="train_only", reviewer=review["reviewer"])


def prepare(source, output, config, exclusions, review_path=None, corrections_path=None):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(output)
    classes = [c for c in config["classes"] if c != "background"]
    if len(classes) != 6:
        raise ValueError("Expected six goal classes")
    drops, review_receipt = reviewed_training_drops(source, config, review_path)
    corrections, correction_receipt = reviewed_training_corrections(source, config, corrections_path)
    if review_path is not None:
        retained = {d["retain_annotation_id"] for d in read_json(review_path)["decisions"]}
        if (drops | retained) & corrections.keys():
            raise ValueError("Corrections conflict with duplicate-review decisions")
    output.mkdir(parents=True)
    seen, receipts = {}, {}
    for split in ("train", "valid"):
        label_file = source / split / "_annotations.coco.json"
        doc = read_json(label_file)
        names = {c["id"]: c["name"] for c in doc["categories"]}
        anns = defaultdict(list)
        for ann in doc["annotations"]:
            anns[ann["image_id"]].append(ann)
        images, counts, excluded = [], Counter(), []
        for im in doc["images"]:
            name = im["file_name"]
            if Path(name).name != name or name.lower().startswith("highbay"):
                raise ValueError("Unexpected path or reserved recording in development data")
            if split == "train" and f"train/{name}" in exclusions:
                excluded.append(name)
                continue
            original = source / split / name
            digest = sha256(original)
            if digest in seen and seen[digest] != split:
                raise ValueError("Cross-split image duplicate")
            seen[digest] = split
            rows = []
            for ann in anns[im["id"]]:
                if split == "train" and ann.get("id") in drops:
                    continue
                label = config["aliases"].get(names[ann["category_id"]])
                if split == "train" and ann.get("id") in corrections:
                    label = corrections[ann["id"]]
                if not label:
                    continue
                box = yolo_box(ann["bbox"], im["width"], im["height"])
                rows.append(f"{classes.index(label)} " + " ".join(f"{n:.9f}" for n in box))
                counts[label] += 1
            image_dest = output / "images" / split / name
            label_dest = output / "labels" / split / (Path(name).stem + ".txt")
            image_dest.parent.mkdir(parents=True, exist_ok=True)
            label_dest.parent.mkdir(parents=True, exist_ok=True)
            image_dest.hardlink_to(original)
            label_dest.write_text("\n".join(rows) + ("\n" if rows else ""))
            images.append(dict(file=name, sha256=digest, labels_sha256=sha256(label_dest)))
        if set(counts) != set(classes):
            raise ValueError("Missing goal class")
        receipts[split] = dict(frames=len(images), counts=dict(counts), images=images,
                               excluded=excluded, annotation_sha256=sha256(label_file))
    dataset = dict(path=str(output.resolve()), train="images/train", val="images/valid",
                   names=dict(enumerate(classes)))
    (output / "dataset.yaml").write_text(yaml.safe_dump(dataset, sort_keys=False))
    report = dict(classes=classes, splits=receipts, test_exported=False, test_evaluated=False,
                  deployment_approved=False, source=str(source.resolve()),
                  dataset_yaml_sha256=sha256(output / "dataset.yaml"),
                  training_label_review=review_receipt,
                  training_label_corrections=correction_receipt,
                  qualification="Upstream labels with known training-source exclusion and optional "
                  "checksum-bound visual corrections; no new boxes. "
                  "Missing annotations and correlated filename-derived splits remain limitations. "
                  "Retains tiny positive boxes. Research training, not competition qualification.")
    write_json(output / "receipt.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--training-review", help="Optional checksum-bound train-only duplicate review")
    parser.add_argument("--label-corrections", help="Optional checksum-bound train-only visual corrections")
    args = parser.parse_args()
    cfg = read_json("configs/goal.json")
    review = read_json("configs/proposal-negative-review-20261005.json")
    report = prepare("data/roboflow-dtr-v10-grouped/coco", args.output, cfg,
                     review["goal"]["exclude_training_sources"], args.training_review,
                     args.label_corrections)
    print({k: {f: v for f, v in r.items() if f != "images"} for k, r in report["splits"].items()})


if __name__ == "__main__":
    main()
