"""Small source-disjoint real-photo development set; never a flight benchmark."""
import argparse
import hashlib
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, validate, write_json
from dtr.vision import RED_BLUE_CLASSES
from scripts.build_public_balloon import ARCHIVE_SHA, INDEX_SHA


def split_assignments(index, review, plan, classes):
    by_id = {row["id"]: row for row in index}
    val_sources = {by_id[i]["source"] for i in plan["validation_source_anchors"]}
    selected = []
    for label in classes:
        for i in review[label]:
            if not by_id[i]["source"].startswith("balloon/train/"):
                raise ValueError("Development pool must use upstream train sources")
            split = "val" if by_id[i]["source"] in val_sources else "train"
            selected.append((i, label, split))
        for i in plan["reserved_upstream_val"][label]:
            if not by_id[i]["source"].startswith("balloon/val/"):
                raise ValueError("Reserved test must use upstream val sources")
            selected.append((i, label, "test"))
    if len({i for i, _, _ in selected}) != len(selected):
        raise ValueError("Duplicate annotation ID")
    return selected


def build(output, archive_path, index_path, review_path, plan_path, config_path):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if sha256(archive_path) != ARCHIVE_SHA or sha256(index_path) != INDEX_SHA:
        raise ValueError("Source archive/index checksum changed")
    index, review, plan, config = map(read_json, (index_path, review_path, plan_path, config_path))
    if set(config["classes"]) != RED_BLUE_CLASSES:
        raise ValueError("Require corrected red/blue taxonomy")
    selected = split_assignments(index, review, plan, config["classes"])
    by_id = {row["id"]: row for row in index}
    output.mkdir(parents=True)
    rows, source_splits = [], {}
    with zipfile.ZipFile(archive_path) as archive:
        for i, label, split in selected:
            item = by_id[i]
            with Image.open(io.BytesIO(archive.read(item["source"]))) as opened:
                image = opened.convert("RGB")
            source_hash = hashlib.sha256(image.tobytes()).hexdigest()
            if source_hash in source_splits and source_splits[source_hash] != split:
                raise ValueError("Cross-split source pixels")
            source_splits[source_hash] = split
            # Holdouts retain one upstream-box crop each. Train adds a low-res,
            # padded context view from the SAME source; not an independent example.
            crops = {"source": image.crop(item["box_xyxy"])}
            if split == "train":
                scan = cv2.resize(np.asarray(image), (320, 240), interpolation=cv2.INTER_AREA)
                box = [int(round(v*(320/image.width if k % 2 == 0 else 240/image.height)))
                       for k, v in enumerate(item["box_xyxy"])]
                a, b, c, d = box
                pad = int(max(c-a, d-b)*.12)
                if c > a and d > b:
                    crops["scan-context"] = Image.fromarray(scan).crop(
                        (max(0, a-pad), max(0, b-pad), min(320, c+pad), min(240, d+pad)))
            for view, crop in crops.items():
                if min(crop.size) < 4:
                    continue
                relative = f"images/{split}/{i:03d}-{view}.png"
                path = output / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                crop.save(path)
                rows.append(dict(path=relative, label=label, split=split,
                                 session=f"photo:{source_hash}", sha256=sha256(path),
                                 source_image=item["source"], source_sha256=source_hash,
                                 annotation_id=i, box_xyxy=item["box_xyxy"], view=view,
                                 label_origin="AI reviewed color, upstream balloon polygon"))
    qualification = dict(domain="generic_real_balloon_photographs_NOT_DTR",
                         label_review="AI reviewed, not human certified",
                         grouping="source-pixel disjoint; recording-session independence UNVERIFIED",
                         validation="development only; previously inspected sources",
                         negatives="other-color balloons only; red/blue non-balloon clutter missing",
                         test="tiny off-domain holdout; insufficient for flight qualification",
                         competition_accuracy_established=False, redistribution_approved=False)
    doc = dict(task="balloon", classes=config["classes"], synthetic=False,
               training_approved=True, deployment_approved=False,
               qualification=qualification, split_provenance=plan,
               source=review["source_url"], source_archive_sha256=ARCHIVE_SHA,
               source_index_sha256=INDEX_SHA, review_sha256=sha256(review_path),
               split_plan_sha256=sha256(plan_path), samples=rows)
    manifest = output / "manifest.json"
    write_json(manifest, doc)
    validate(manifest, config)
    receipt = dict(manifest_sha256=sha256(manifest), samples=len(rows),
                   counts=dict(sorted(Counter(f"{r['split']}/{r['label']}" for r in rows).items())),
                   distinct_source_images=dict(Counter(source_splits.values())),
                   test_evaluated=False, deployment_approved=False, qualification=qualification)
    write_json(output / "receipt.json", receipt)
    print(json.dumps(receipt, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    build(args.output, "data/raw/matterport-balloon/balloon_dataset.zip",
          "data/raw/matterport-balloon/review/index.json",
          "configs/balloon-red-blue-public-review.json",
          "configs/balloon-red-blue-bootstrap-splits.json", "configs/balloon-red-blue.json")


if __name__ == "__main__":
    main()
