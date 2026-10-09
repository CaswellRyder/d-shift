"""Quarantine localization-quality examples from reviewed TRAIN balloon anchors.

IoU is supervision for box quality, not a new class label or qualification score.
Unmatched proposals are unknown, never automatic negatives. Manual/AI admission
is required after inspecting crops and the frames' incomplete annotations.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from dtr.data import read_json, sha256, write_json
from dtr.tracking import iou
from scripts.audit_public_balloon_export import bounded_read
from scripts.build_indoor_balloon_training import admitted
from scripts.prepare_indoor_balloon_review import ARCHIVE_SHA, MAP, family
from scripts.research_balloon_search import experimental


def quality_target(candidate, positives, negatives):
    overlaps = [(iou(candidate["box"], p["target_box"]), p) for p in positives]
    best, anchor = max(overlaps, default=(0., None), key=lambda pair: pair[0])
    if best >= .02:
        return dict(quality=best, anchor_id=anchor["id"], kind="reviewed_balloon_overlap")
    # Only an already reviewed negative crop can supervise an unmatched candidate.
    exact = next((n for n in negatives if candidate["crop_box"] == n["crop_box"]), None)
    if exact is not None:
        return dict(quality=0., anchor_id=exact["id"], kind="exact_reviewed_background")
    return None


def prepare(manifest, queue_root, review, archive_path, output):
    root, output = Path(queue_root).resolve(), Path(output)
    if output.exists():
        raise FileExistsError(output)
    queue = read_json(root / "review.json")
    decision = read_json(review)
    if decision.get("training_approved") is False or decision.get("deployment_approved") is not False:
        raise ValueError("Require training-only reviewed anchors")
    anchors = admitted(root, review, manifest)
    if queue.get("archive_sha256") != ARCHIVE_SHA or sha256(archive_path) != ARCHIVE_SHA:
        raise ValueError("Training archive changed")
    with zipfile.ZipFile(archive_path) as archive:
        doc = json.loads(bounded_read(archive, "train/_annotations.coco.json", 50_000_000))
    annotations = {r["id"]: r for r in doc["annotations"]}
    sources = {r["id"]: r for r in doc["images"]}
    classes = {r["id"]: r["name"] for r in doc["categories"]}
    output.mkdir(parents=True)
    rows, frames = [], []
    for frame in queue["frames"]:
        path = (root / frame["path"]).resolve()
        if (not path.is_relative_to(root) or sha256(path) != frame["sha256"]
                or family(Path(frame["source"]).name) is None):
            raise ValueError("Invalid training frame path, pixels or family")
        with Image.open(path) as im:
            rgb = np.asarray(im.convert("RGB"))
        if rgb.shape != (240, 320, 3):
            raise ValueError("Require reviewed 320x240 frame")
        positives, negatives = [], []
        for row in (r for r in anchors if r["frame_id"] == frame["id"]):
            if row["label"] == "background":
                negatives.append(row)
                continue
            annotation = annotations[row["annotation_id"]]
            source = sources[annotation["image_id"]]
            if ("train/" + source["file_name"] != row["source_image"]
                    or MAP.get(classes[annotation["category_id"]]) != row["label"]):
                raise ValueError("Reviewed anchor annotation identity changed")
            x, y, w, h = annotation["bbox"]
            box = [x*320/source["width"], y*240/source["height"],
                   (x+w)*320/source["width"], (y+h)*240/source["height"]]
            positives.append(dict(row, target_box=box))
        ids = []
        for candidate in experimental(rgb, "mser"):
            target = quality_target(candidate, positives, negatives)
            if target is None:
                continue
            a, b, c, d = candidate["crop_box"]
            relative = f"crops/{len(rows):03d}.png"
            dest = output / relative
            dest.parent.mkdir(exist_ok=True)
            Image.fromarray(rgb[b:d, a:c]).save(dest)
            anchor = next(r for r in anchors if r["id"] == target["anchor_id"])
            ids.append(len(rows))
            rows.append(dict(candidate, **target, id=len(rows), path=relative, sha256=sha256(dest),
                frame_id=frame["id"], split="train", session=anchor["session"],
                source_image=anchor["source_image"], source_sha256=anchor["source_sha256"],
                source_group=anchor["source_group"],
                target_boxes=[p["target_box"] for p in positives]))
        relative = f"frames/{frame['id']:03d}.png"
        dest = output / relative
        dest.parent.mkdir(exist_ok=True)
        Image.fromarray(rgb).save(dest)
        frames.append(dict(frame, path=relative, sha256=sha256(dest), crop_ids=ids,
                           quality_targets=[dict(label=p["label"], box=p["target_box"]) for p in positives]))
    report = dict(samples=rows, frames=frames, base_manifest_sha256=sha256(manifest),
        parent_queue_sha256=sha256(root / "review.json"), parent_review_sha256=sha256(review),
        archive_sha256=ARCHIVE_SHA, script_sha256=sha256(__file__),
        search_sha256=sha256("scripts/research_balloon_search.py"),
        training_approved=False, review_required=True, test_evaluated=False, deployment_approved=False,
        scope="TRAIN-only proposal IoU targets from admitted anchors; incomplete frame truth requires review",
        counts=dict(Counter(r["kind"] for r in rows)))
    write_json(output / "review.json", report)
    for start in range(0, len(rows), 48):
        sheet = Image.new("RGB", (960, 720), "#ddd")
        draw = ImageDraw.Draw(sheet)
        for n, row in enumerate(rows[start:start+48]):
            x, y = n % 8 * 120, n // 8 * 120
            with Image.open(output / row["path"]) as im:
                sheet.paste(ImageOps.contain(im, (116, 92)), (x, y+24))
            draw.text((x, y), f"{row['id']} q{row['quality']:.2f} f{row['frame_id']}", fill="black")
        sheet.save(output / f"crops-{start:03d}.jpg")
    for start in range(0, len(frames), 8):
        sheet = Image.new("RGB", (1280, 530), "#ddd")
        draw = ImageDraw.Draw(sheet)
        for n, frame in enumerate(frames[start:start+8]):
            x, y = n % 4*320, n // 4*265
            with Image.open(output / frame["path"]) as im:
                marked = im.copy()
            marker = ImageDraw.Draw(marked)
            for target in frame["quality_targets"]:
                marker.rectangle(target["box"], outline="yellow", width=1)
            sheet.paste(marked, (x, y+24))
            draw.text((x, y), f"frame {frame['id']}", fill="black")
        sheet.save(output / f"frames-{start:03d}.jpg")
    return dict(rows=len(rows), counts=report["counts"], sha256=sha256(output / "review.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "queue", "review", "archive", "output"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    print(prepare(args.manifest, args.queue, args.review, args.archive, args.output))
