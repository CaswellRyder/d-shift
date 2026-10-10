"""Reserve a larger, disjoint IMG_ development panel, before any model predictions.

One unaugmented image per unused IMG_#### family from the upstream valid/test splits.
Frames perceptually close to training photos, or to an earlier panel frame, are rejected.
Upstream boxes are provisional truth until a frame-by-frame visual review admits them.
"""
import argparse
from collections import defaultdict
import io
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageDraw

from dtr.data import read_json, sha256, write_json
from scripts.audit_public_balloon_export import bounded_read
from scripts.prepare_balloon_scene_review import development_family
from scripts.prepare_indoor_balloon_review import ARCHIVE_SHA, MAP, fingerprint, holdout_distance


def training_fingerprints(original_manifest, training_queues):
    base = read_json(original_manifest)
    old_zip = Path("data/raw/matterport-balloon/balloon_dataset.zip")
    if sha256(old_zip) != base["source_archive_sha256"]:
        raise ValueError("Original archive changed")
    hashes = []
    with zipfile.ZipFile(old_zip) as old:
        for source in sorted({r["source_image"] for r in base["samples"] if r["split"] == "train"}):
            with Image.open(io.BytesIO(old.read(source))) as im:
                hashes.append(fingerprint(np.asarray(im.convert("RGB"))))
    for queue_root in map(Path, training_queues):
        queue = read_json(queue_root / "review.json")
        if queue.get("training_approved") is not False or queue["archive_sha256"] != ARCHIVE_SHA:
            raise ValueError("Unexpected training queue")
        for row in queue["frames"]:
            path = queue_root / row["path"]
            if sha256(path) != row["sha256"]:
                raise ValueError("Training frame changed")
            if development_family(row["source"]):
                raise ValueError("Training queue contains the reserved IMG_ family")
            with Image.open(path) as im:
                hashes.append(fingerprint(np.asarray(im.convert("RGB"))))
    return hashes


def prior_panel(exclude_queues):
    families, hashes = set(), []
    for queue_root in map(Path, exclude_queues):
        queue = read_json(queue_root / "review.json")
        for row in queue["frames"]:
            families.add(row["source_group"])
            path = queue_root / row["path"]
            if sha256(path) != row["sha256"]:
                raise ValueError("Prior panel frame changed")
            with Image.open(path) as im:
                hashes.append(fingerprint(np.asarray(im.convert("RGB"))))
        families.update(r["source_group"] for r in queue.get("rejected", []) if "source_group" in r)
    return families, hashes


def prepare(root, original_manifest, training_queues, exclude_queues, splits, limit, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    archive_path = Path(root) / "dataset.zip"
    if sha256(archive_path) != ARCHIVE_SHA:
        raise ValueError("Archive changed")
    training = training_fingerprints(original_manifest, training_queues)
    used, panel = prior_panel(exclude_queues)
    frames, rejected = [], []
    with zipfile.ZipFile(archive_path) as archive:
        groups = {}
        for split in splits:
            doc = json.loads(bounded_read(archive, f"{split}/_annotations.coco.json", 50_000_000))
            categories = {r["id"]: r["name"] for r in doc["categories"]}
            by_image = defaultdict(list)
            for a in doc["annotations"]:
                by_image[a["image_id"]].append(a)
            for row in sorted(doc["images"], key=lambda r: r["file_name"]):
                family = development_family(row["file_name"])
                labels = {categories[a["category_id"]] for a in by_image[row["id"]]}
                if family and family not in used and labels <= MAP.keys():
                    groups.setdefault(family, (split, row, [
                        dict(label=MAP[categories[a["category_id"]]], bbox=a["bbox"])
                        for a in by_image[row["id"]]]))
        eligible = [groups[f] for f in sorted(groups)]
        output.mkdir(parents=True)
        (output / "frames").mkdir()
        for i in np.linspace(0, len(eligible) - 1, min(limit, len(eligible)), dtype=int):
            split, row, annotations = eligible[int(i)]
            family = development_family(row["file_name"])
            raw = bounded_read(archive, f"{split}/{row['file_name']}", 20_000_000)
            with Image.open(io.BytesIO(raw)) as im:
                rgb = np.asarray(im.convert("RGB"))
            distance = holdout_distance(rgb, training)
            nearest_panel = holdout_distance(rgb, panel) if panel else None
            if distance <= 8 or (nearest_panel is not None and nearest_panel <= 8):
                rejected.append(dict(source=row["file_name"], source_group=family,
                                     nearest_training_hamming=distance,
                                     nearest_panel_hamming=nearest_panel))
                continue
            panel.append(fingerprint(rgb))
            scan = cv2.resize(rgb, (320, 240), interpolation=cv2.INTER_AREA)
            path = f"frames/{len(frames):03d}.png"
            Image.fromarray(scan).save(output / path)
            sx, sy = 320 / rgb.shape[1], 240 / rgb.shape[0]
            truth = [dict(label=a["label"], box=[a["bbox"][0]*sx, a["bbox"][1]*sy,
                                                 (a["bbox"][0]+a["bbox"][2])*sx,
                                                 (a["bbox"][1]+a["bbox"][3])*sy])
                     for a in annotations]
            frames.append(dict(id=len(frames), path=path, sha256=sha256(output / path),
                               source=row["file_name"], upstream_split=split, source_group=family,
                               nearest_training_hamming=distance, nearest_panel_hamming=nearest_panel,
                               provisional_truth=truth))
    report = dict(frames=frames, rejected=rejected, archive_sha256=ARCHIVE_SHA,
                  script_sha256=sha256(__file__), base_manifest_sha256=sha256(original_manifest),
                  training_queue_sha256=[sha256(Path(q) / "review.json") for q in training_queues],
                  excluded_queue_sha256=[sha256(Path(q) / "review.json") for q in exclude_queues],
                  splits=list(splits), eligible_families=len(eligible), limit=limit,
                  evaluation_approved=False, training_approved=False, predictions_run=False,
                  deployment_approved=False,
                  scope="Second development panel from the reserved IMG_ family; NOT final test "
                        "or representative Pi recordings",
                  caveat="Consecutive IMG_ numbers are likely the same sessions and rooms; filename "
                         "families and perceptual screening do not verify independence")
    write_json(output / "review.json", report)
    for start in range(0, len(frames), 8):
        sheet = Image.new("RGB", (1280, 530), "#ddd")
        draw = ImageDraw.Draw(sheet)
        for n, row in enumerate(frames[start:start+8]):
            x, y = n % 4 * 320, n // 4 * 265
            with Image.open(output / row["path"]) as im:
                draw_frame = ImageDraw.Draw(im)
                for j, t in enumerate(row["provisional_truth"]):
                    draw_frame.rectangle(t["box"], outline="lime", width=1)
                    draw_frame.text((t["box"][0], t["box"][1]), f"{j}:{t['label'][:1]}", fill="lime")
                sheet.paste(im, (x, y+24))
            draw.text((x, y), f"frame {row['id']} {row['source_group']}", fill="black")
        sheet.save(output / f"frames-{start:03d}.jpg")
    print(dict(frames=len(frames), rejected=len(rejected), eligible=len(eligible),
               queue_sha256=sha256(output / "review.json")))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--training-queue", action="append", required=True)
    p.add_argument("--exclude-queue", action="append", default=[])
    p.add_argument("--split", action="append", choices=("valid", "test"))
    p.add_argument("--limit", type=int, default=96)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    prepare(args.root, args.manifest, args.training_queue, args.exclude_queue,
            args.split or ["valid", "test"], args.limit, args.output)
