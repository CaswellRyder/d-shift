"""Full-frame proposal coverage on reviewed training photos, NOT detector accuracy."""
import argparse
import hashlib
import io
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, validate
from dtr.tracking import iou
from dtr.vision import RED_BLUE_CLASSES, RED_BLUE_PROFILE, proposals


def coverage(box, label, candidates):
    """Color-group agreement is proposal evidence, never a classified identity."""
    group = {"red_balloon": 0, "blue_balloon": 1}[label]
    overlap = max((iou(box, c["box"]) for c in candidates
                   if c["color_group"] == group), default=0.)
    return dict(best_iou=float(overlap), covered=overlap >= .5)


def audit(manifest, archive_path):
    doc = read_json(manifest)
    if (set(doc["classes"]) != RED_BLUE_CLASSES
            or not doc.get("qualification", {}).get("training_only")):
        raise ValueError("Requires a red/blue training-only seed")
    validate(manifest, dict(task="balloon", classes=doc["classes"]), splits=("train",))
    if sha256(archive_path) != doc["source_archive_sha256"]:
        raise ValueError("Source archive changed")
    grouped = defaultdict(list)
    for row in doc["samples"]:
        if row["split"] != "train":
            raise ValueError("Only training images may be audited")
        if row["label"] != "background":
            grouped[row["source_image"]].append(row)
    results = []
    with zipfile.ZipFile(archive_path) as archive:
        for source, rows in sorted(grouped.items()):
            with Image.open(io.BytesIO(archive.read(source))) as opened:
                rgb = np.asarray(opened.convert("RGB"))
            digest = hashlib.sha256(rgb.tobytes()).hexdigest()
            if any(r["source_sha256"] != digest for r in rows):
                raise ValueError("Source pixels changed")
            scan = cv2.resize(rgb, (320, 240), interpolation=cv2.INTER_AREA)
            found = proposals(scan, "balloon", limit=12, profile=RED_BLUE_PROFILE)
            for row in rows:
                box = [v * (320/rgb.shape[1] if i % 2 == 0 else 240/rgb.shape[0])
                       for i, v in enumerate(row["box_xyxy"])]
                results.append(dict(annotation_id=row["annotation_id"], source=source,
                                    label=row["label"], scan_box=box, candidates=len(found),
                                    **coverage(box, row["label"], found)))
    counts = {}
    for label in ("red_balloon", "blue_balloon"):
        selected = [r for r in results if r["label"] == label]
        counts[label] = dict(covered=sum(r["covered"] for r in selected), total=len(selected))
    return dict(scope="selected off-domain training targets; incomplete scene labels",
                scan_size=[320, 240], proposal_limit=12, profile=RED_BLUE_PROFILE,
                overlap_threshold=.5, classification_measured=False, pi_timing_measured=False,
                precision_measured=False, deployment_approved=False,
                manifest_sha256=sha256(manifest),
                vision_sha256=sha256(Path(__file__).resolve().parents[1]/"src/dtr/vision.py"),
                counts=counts, results=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--archive", default="data/raw/matterport-balloon/balloon_dataset.zip")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = audit(args.manifest, args.archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
