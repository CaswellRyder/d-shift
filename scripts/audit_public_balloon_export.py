"""Inspect public COCO metadata and a bounded train-only preview; no training admission."""
import argparse
from collections import Counter, defaultdict
import io
import json
from pathlib import Path, PurePosixPath
import re

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import zipfile

from dtr.data import read_json, sha256, write_json


def source_group(filename):
    # Roboflow siblings typically preserve the original basename before .rf.HASH.
    # Heuristic only: content/perceptual dedup and recording-group review still needed.
    return re.sub(r"\.rf\.[a-fA-F0-9]+(?=\.)", "", PurePosixPath(filename).name).lower()


def checked_member(path):
    p = PurePosixPath(path)
    if p.is_absolute() or ".." in p.parts or "\\" in path:
        raise ValueError("Unsafe archive member")
    return p


def bounded_read(archive, name, limit):
    checked_member(name)
    if archive.getinfo(name).file_size > limit:
        raise ValueError("Archive member exceeds read bound")
    return archive.read(name)


def audit(root, output):
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError(output)
    receipt = read_json(root / "download-receipt.json")
    archive_path = root / "dataset.zip"
    if sha256(archive_path) != receipt["archive_sha256"]:
        raise ValueError("Archive changed")
    totals, groups, previews = {}, defaultdict(set), []
    rng = np.random.default_rng(20261008)
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive members")
        annotations = sorted(n for n in names if n.endswith("/_annotations.coco.json"))
        if not annotations:
            raise ValueError("No split COCO annotations found")
        for name in annotations:
            split = checked_member(name).parent.name
            if split not in ("train", "valid", "test") or split in totals:
                raise ValueError("Unexpected or duplicate split")
            doc = json.loads(bounded_read(archive, name, 50_000_000))
            categories = {r["id"]: r["name"] for r in doc["categories"]}
            images = {r["id"]: r for r in doc["images"]}
            if len(images) != len(doc["images"]):
                raise ValueError("Duplicate image IDs")
            by_image, counts = defaultdict(list), Counter()
            for a in doc["annotations"]:
                if a["image_id"] not in images or a["category_id"] not in categories:
                    raise ValueError("Dangling annotation reference")
                label = categories[a["category_id"]]
                counts[label] += 1
                by_image[a["image_id"]].append(dict(label=label, bbox=a["bbox"]))
            for row in images.values():
                checked_member(row["file_name"])
                groups[source_group(row["file_name"])].add(split)
            totals[split] = dict(images=len(images), annotations=sum(counts.values()),
                                 classes=categories, class_counts=dict(counts),
                                 without_annotations=sum(i not in by_image for i in images),
                                 basename_groups=len({source_group(r["file_name"]) for r in images.values()}))
            if split != "train":
                continue  # Holdout metadata only; no validation/test pixels opened.
            chosen = set()
            for label in sorted(counts):
                eligible = [i for i in images if any(a["label"] == label for a in by_image[i])]
                rng.shuffle(eligible)
                chosen.update(eligible[:6])
            for ident in sorted(chosen):
                row = images[ident]
                name_in_zip = str(PurePosixPath(name).parent / row["file_name"])
                raw = bounded_read(archive, name_in_zip, 20_000_000)
                with Image.open(io.BytesIO(raw)) as opened:
                    if opened.width*opened.height > 16_000_000:
                        raise ValueError("Preview dimensions exceed bound")
                    image = opened.convert("RGB")
                if image.size != (row["width"], row["height"]):
                    raise ValueError("COCO dimensions differ from image")
                preview = ImageOps.contain(image, (380, 340))
                draw = ImageDraw.Draw(preview)
                for a in by_image[ident]:
                    x, y, w, h = a["bbox"]
                    sx, sy = preview.width/image.width, preview.height/image.height
                    box = [x*sx, y*sy, (x+w)*sx, (y+h)*sy]
                    draw.rectangle(box, outline="lime", width=2)
                    draw.text((box[0], max(0, box[1]-12)), a["label"], fill="lime", stroke_width=1, stroke_fill="black")
                previews.append((row, preview, by_image[ident]))
    output.mkdir(parents=True)
    for start in range(0, len(previews), 12):
        sheet = Image.new("RGB", (1200, 1080), "#ddd")
        draw = ImageDraw.Draw(sheet)
        for i, (_, preview, _) in enumerate(previews[start:start+12]):
            x, y = i % 3*400, i // 3*270
            tile = ImageOps.contain(preview, (396, 246))
            sheet.paste(tile, (x, y+22))
            draw.text((x, y), str(start+i), fill="black")
        sheet.save(output / f"train-preview-{start:03d}.jpg")
    report = dict(source=receipt["source"], archive_sha256=receipt["archive_sha256"],
                  script_sha256=sha256(__file__), split_metadata=totals,
                  cross_split_basename_groups={k:sorted(v) for k,v in groups.items() if len(v)>1},
                  grouping_caveat="Filename heuristic only; no content dedup or verified recording sessions",
                  previews=[dict(id=i, file=row["file_name"], annotations=anns)
                            for i,(row,_,anns) in enumerate(previews)],
                  preview_scope="Training pixels only; upstream valid/test metadata inspected, pixels not opened",
                  training_approved=False, deployment_approved=False,
                  review_required=True, label_mapping_approved=False)
    write_json(output / "audit.json", report)
    print(dict(splits=totals, previews=len(previews),
               cross_split_basename_groups=len(report["cross_split_basename_groups"]), training_approved=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    audit(args.root, args.output)
