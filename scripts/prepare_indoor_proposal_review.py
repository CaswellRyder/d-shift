"""Create review-only detector-shaped positives from admitted indoor TRAIN frames."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from dtr.data import read_json, sha256, write_json
from dtr.tracking import iou
from scripts.audit_public_balloon_export import bounded_read
from scripts.build_indoor_balloon_training import admitted
from scripts.build_red_blue_proposal_positives import matched_target
from scripts.prepare_indoor_balloon_review import ARCHIVE_SHA, MAP
from scripts.research_balloon_search import experimental


def prepare(root, review, manifest, archive_path, output):
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError(output)
    anchors = admitted(root, review, manifest)
    anchors = [r for r in anchors if r["label"] != "background"]
    queue = read_json(root / "review.json")
    if sha256(archive_path) != ARCHIVE_SHA:
        raise ValueError("Archive changed")
    with zipfile.ZipFile(archive_path) as archive:
        doc = json.loads(bounded_read(archive, "train/_annotations.coco.json", 50_000_000))
    annotations = {r["id"]: r for r in doc["annotations"]}
    sources = {r["id"]: r for r in doc["images"]}
    classes = {r["id"]: r["name"] for r in doc["categories"]}
    output.mkdir(parents=True)
    (output / "crops").mkdir()
    frames, rows, seen = [], [], set()
    for frame in queue["frames"]:
        path = root / frame["path"]
        if sha256(path) != frame["sha256"]:
            raise ValueError("Source frame changed")
        with Image.open(path) as im:
            rgb = np.asarray(im.convert("RGB"))
        if rgb.shape != (240, 320, 3):
            raise ValueError("Require reviewed scan coordinates")
        targets = []
        for row in anchors:
            if row["frame_id"] != frame["id"]:
                continue
            a = annotations[row["annotation_id"]]
            source = sources[a["image_id"]]
            if ("train/" + source["file_name"] != row["source_image"]
                    or MAP.get(classes[a["category_id"]]) != row["label"]):
                raise ValueError("Annotation identity changed")
            x, y, w, h = a["bbox"]
            box = [x*320/source["width"], y*240/source["height"],
                   (x+w)*320/source["width"], (y+h)*240/source["height"]]
            targets.append(dict(row, scan_box=box))
        ids = []
        for variant in ("baseline", "mser"):
            for candidate in experimental(rgb, variant):
                target = matched_target(candidate, targets)
                if target is None:
                    continue
                x, y, z, w = candidate["crop_box"]
                crop = rgb[y:w, x:z]
                key = (target["source_image"], crop.shape, hashlib.sha256(crop.tobytes()).hexdigest())
                if key in seen or candidate["crop_box"] == target["crop_box"]:
                    continue
                seen.add(key)
                relative = f"crops/{len(rows):03d}.png"
                Image.fromarray(crop).save(output / relative)
                ids.append(len(rows))
                rows.append(dict(target, id=len(rows), path=relative, sha256=sha256(output / relative),
                                 anchor_id=target["id"], proposal_variant=variant,
                                 box=candidate["box"], crop_box=candidate["crop_box"],
                                 match_iou=iou(candidate["box"], target["scan_box"])))
        relative = frame["path"]
        (output / relative).parent.mkdir(exist_ok=True)
        Image.fromarray(rgb).save(output / relative)
        frames.append(dict(frame, crop_ids=ids, sha256=sha256(output / relative)))
    report = dict(base_manifest_sha256=sha256(manifest), parent_queue_sha256=sha256(root / "review.json"),
                  parent_review_sha256=sha256(review), archive_sha256=ARCHIVE_SHA,
                  samples=rows, frames=frames, script_sha256=sha256(__file__),
                  search_script_sha256=sha256("scripts/research_balloon_search.py"),
                  training_approved=False, review_required=True, deployment_approved=False,
                  scope="Additional proposal views of existing training frames; no new independent data",
                  counts=dict(Counter(r["label"] for r in rows)))
    write_json(output / "review.json", report)
    for start in range(0, len(rows), 48):
        sheet = Image.new("RGB", (960, 720), "#ddd")
        draw = ImageDraw.Draw(sheet)
        for n, row in enumerate(rows[start:start+48]):
            x, y = n % 8 * 120, n // 8 * 120
            with Image.open(output / row["path"]) as im:
                sheet.paste(ImageOps.contain(im, (116, 92)), (x, y+24))
            draw.text((x, y), f"{row['id']} {row['label'][:1]} f{row['frame_id']}", fill="black")
        sheet.save(output / f"crops-{start:03d}.jpg")
    print(dict(crops=len(rows), counts=report["counts"], queue_sha256=sha256(output / "review.json")))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("queue", "review", "manifest", "archive", "output"):
        p.add_argument(f"--{name}", required=True)
    args = p.parse_args()
    prepare(args.queue, args.review, args.manifest, args.archive, args.output)
