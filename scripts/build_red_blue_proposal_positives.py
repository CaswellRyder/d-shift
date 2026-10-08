"""Train-only proposal crops inheriting reviewed labels through strict box matches."""
import argparse
from collections import Counter, defaultdict
import hashlib
import io
from pathlib import Path
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from dtr.data import read_json, sha256, write_json
from dtr.tracking import iou
from scripts.research_balloon_search import experimental

LABELS = ("red_balloon", "blue_balloon")


def matched_target(candidate, targets):
    matches = [t for t in targets if iou(candidate["box"], t["scan_box"]) >= .7]
    if len(matches) != 1:
        return None
    target = matches[0]
    if candidate["color_group"] != LABELS.index(target["label"]):
        return None
    if any(t["annotation_id"] != target["annotation_id"]
           and t["label"] != target["label"]
           and iou(candidate["crop_box"], t["scan_box"]) > .3 for t in targets):
        return None
    return target


def build(manifest, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    doc = read_json(manifest)
    archive_path = Path("data/raw/matterport-balloon/balloon_dataset.zip")
    if doc.get("training_approved") is not True or sha256(archive_path) != doc["source_archive_sha256"]:
        raise ValueError("Require approved, hash-bound source data")
    sources = defaultdict(dict)
    forbidden = {r["source_image"] for r in doc["samples"] if r["split"] != "train"}
    for row in doc["samples"]:
        if row["split"] == "train" and row["label"] in LABELS:
            if row["source_image"] in forbidden:
                raise ValueError("Cross-split source overlap")
            sources[row["source_image"]][row["annotation_id"]] = row
    output.mkdir(parents=True)
    rows, seen = [], set()
    with zipfile.ZipFile(archive_path) as archive:
        for source, annotations in sorted(sources.items()):
            with Image.open(io.BytesIO(archive.read(source))) as im:
                im = im.convert("RGB")
                source_hash = hashlib.sha256(im.tobytes()).hexdigest()
                rgb = cv2.resize(np.asarray(im), (320, 240), interpolation=cv2.INTER_AREA)
                targets = []
                for row in annotations.values():
                    if row["source_sha256"] != source_hash:
                        raise ValueError("Source pixels changed")
                    targets.append(dict(row, scan_box=[v*(320/im.width if i % 2 == 0 else 240/im.height)
                                                       for i, v in enumerate(row["box_xyxy"])]))
            for variant in ("baseline", "mser"):
                for candidate in experimental(rgb, variant):
                    target = matched_target(candidate, targets)
                    if target is None:
                        continue
                    a, b, c, d = candidate["crop_box"]
                    crop = rgb[b:d, a:c]
                    key = (source, hashlib.sha256(crop.tobytes()).hexdigest(), crop.shape)
                    if key in seen:
                        continue
                    seen.add(key)
                    relative = f"crops/{len(rows):03d}.png"
                    path = output / relative
                    path.parent.mkdir(exist_ok=True)
                    Image.fromarray(crop).save(path)
                    rows.append(dict(id=len(rows), path=relative, sha256=sha256(path),
                                     label=target["label"], split="train", session=target["session"],
                                     source_image=source, source_sha256=source_hash,
                                     annotation_id=target["annotation_id"], proposal_variant=variant,
                                     box=candidate["box"], crop_box=candidate["crop_box"],
                                     target_box=target["scan_box"], match_iou=iou(candidate["box"], target["scan_box"])))
    report = dict(base_manifest_sha256=sha256(manifest), archive_sha256=sha256(archive_path),
                  script_sha256=sha256(__file__), samples=rows,
                  label_origin="Reviewed training balloon label, unique same-color proposal IoU >=0.7",
                  scope="Additional views of existing training photos, not independent examples",
                  counts=dict(Counter(r["label"] for r in rows)),
                  review_required=True, training_approved=False, deployment_approved=False)
    write_json(output / "review.json", report)
    for start in range(0, len(rows), 48):
        sheet = Image.new("RGB", (960, 720), "#ddd")
        draw = ImageDraw.Draw(sheet)
        for i, row in enumerate(rows[start:start+48]):
            x, y = i % 8 * 120, i // 8 * 120
            with Image.open(output / row["path"]) as crop:
                sheet.paste(ImageOps.contain(crop, (116, 92)), (x, y+24))
            draw.text((x, y), f"{row['id']} {row['label'][:1]} {row['match_iou']:.2f}", fill="black")
        sheet.save(output / f"sheet-{start:03d}.jpg")
    print(dict(crops=len(rows), counts=report["counts"], queue_sha256=sha256(output / "review.json")))


def checked_positives(root, review_path, manifest):
    """Revalidate admission, source partition, class identity, geometry, and pixels."""
    root = Path(root).resolve()
    queue = read_json(root / "review.json")
    decision = read_json(review_path)
    if (queue["base_manifest_sha256"] != sha256(manifest)
            or decision["queue_sha256"] != sha256(root / "review.json")):
        raise ValueError("Positive queue/review hash mismatch")
    doc = read_json(manifest)
    anchors = {(r["source_image"], r["annotation_id"]): r for r in doc["samples"]
               if r["split"] == "train" and r["label"] in LABELS}
    forbidden = {r["source_image"] for r in doc["samples"] if r["split"] != "train"}
    ids = decision["admit"]
    if not ids or len(ids) != len(set(ids)) or any(type(i) is not int or not 0 <= i < len(queue["samples"]) for i in ids):
        raise ValueError("Invalid positive admission")
    rows = [queue["samples"][i] for i in ids]
    for row in rows:
        anchor = anchors.get((row["source_image"], row["annotation_id"]))
        path = (root / row["path"]).resolve()
        boxes = (row["box"], row["crop_box"], row["target_box"])
        valid_boxes = all(len(box) == 4 and np.isfinite(box).all()
                          and 0 <= box[0] < box[2] <= 320 and 0 <= box[1] < box[3] <= 240 for box in boxes)
        if (not anchor or row["source_image"] in forbidden or row["split"] != "train"
                or any(row[k] != anchor[k] for k in ("label", "session", "source_sha256"))
                or not valid_boxes or iou(row["box"], row["target_box"]) < .7
                or not path.is_relative_to(root) or sha256(path) != row["sha256"]):
            raise ValueError("Invalid positive source, geometry or crop")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    build(args.manifest, args.output)
