"""Reserve a lockbox IMG_ development panel from the upstream train split, before any predictions.

Upstream train images exist only as three Roboflow-augmented versions per source
(50% horizontal flip, 50% vertical flip, blur up to 1.5 px, 0.5% salt-and-pepper
noise). One version per unused IMG_#### family is kept, the first by filename.
The visual review records upside-down frames so the bundle can flip them back.
Frames perceptually close to training photos, or to any earlier panel, are rejected.
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

from dtr.data import sha256, write_json
from scripts.audit_public_balloon_export import bounded_read
from scripts.prepare_balloon_scene_review import development_family
from scripts.prepare_development_panel import prior_panel, training_fingerprints
from scripts.prepare_indoor_balloon_review import ARCHIVE_SHA, MAP, fingerprint, holdout_distance


def eligible_families(archive, used, split="train"):
    doc = json.loads(bounded_read(archive, f"{split}/_annotations.coco.json", 50_000_000))
    categories = {r["id"]: r["name"] for r in doc["categories"]}
    by_image = defaultdict(list)
    for a in doc["annotations"]:
        by_image[a["image_id"]].append(a)
    versions = defaultdict(list)
    for row in sorted(doc["images"], key=lambda r: r["file_name"]):
        family = development_family(row["file_name"])
        if family and family not in used:
            versions[family].append(row)
    groups = {}
    for family, rows in versions.items():
        labels = {categories[a["category_id"]] for r in rows for a in by_image[r["id"]]}
        if labels <= MAP.keys():
            row = rows[0]
            groups[family] = (row, len(rows), [dict(label=MAP[categories[a["category_id"]]], bbox=a["bbox"])
                                               for a in by_image[row["id"]]])
    return [groups[f] for f in sorted(groups)]


def prepare(root, original_manifest, training_queues, exclude_queues, limit, output):
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
        eligible = eligible_families(archive, used)
        output.mkdir(parents=True)
        (output / "frames").mkdir()
        for i in np.linspace(0, len(eligible) - 1, min(limit, len(eligible)), dtype=int):
            row, version_count, annotations = eligible[int(i)]
            family = development_family(row["file_name"])
            raw = bounded_read(archive, f"train/{row['file_name']}", 20_000_000)
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
                               source=row["file_name"], upstream_split="train", source_group=family,
                               upstream_versions=version_count,
                               nearest_training_hamming=distance, nearest_panel_hamming=nearest_panel,
                               provisional_truth=truth))
    report = dict(frames=frames, rejected=rejected, archive_sha256=ARCHIVE_SHA,
                  script_sha256=sha256(__file__), base_manifest_sha256=sha256(original_manifest),
                  training_queue_sha256=[sha256(Path(q) / "review.json") for q in training_queues],
                  excluded_queue_sha256=[sha256(Path(q) / "review.json") for q in exclude_queues],
                  splits=["train"], eligible_families=len(eligible), limit=limit,
                  evaluation_approved=False, training_approved=False, predictions_run=False,
                  deployment_approved=False,
                  scope="Lockbox development panel from the reserved IMG_ family (upstream train "
                        "split); NOT final test or representative Pi recordings",
                  caveat="Frames are Roboflow-augmented versions (flips, blur, noise); consecutive "
                         "IMG_ numbers are likely the same sessions and rooms as earlier panels")
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
    p.add_argument("--limit", type=int, default=96)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    prepare(args.root, args.manifest, args.training_queue, args.exclude_queue, args.limit, args.output)
