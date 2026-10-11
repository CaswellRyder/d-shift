"""Trace missed admitted training targets through MSER; no model or holdout tuning."""
import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageDraw

from dtr.data import read_json, sha256, write_json
from dtr.tracking import iou
from scripts.audit_public_balloon_export import bounded_read
from scripts.build_indoor_balloon_training import admitted
from scripts.prepare_indoor_balloon_review import ARCHIVE_SHA
from scripts.research_balloon_search import candidate, experimental, select


def training_source(source):
    if Path(source).name.lower().startswith("img_"):
        raise ValueError("Reserved IMG family cannot enter training audit")


def trace(rgb):
    if rgb.shape != (240, 320, 3) or rgb.dtype != np.uint8:
        raise ValueError("Require 320x240 RGB uint8")
    stages = dict(raw=[], contrast=[], geometry=[])
    planes = (cv2.subtract(rgb[:, :, 0], cv2.max(rgb[:, :, 1], rgb[:, :, 2])),
              cv2.subtract(rgb[:, :, 2], cv2.max(rgb[:, :, 0], rgb[:, :, 1])))
    for group, plane in enumerate(planes):
        regions, _ = cv2.MSER_create(5, 6, 25000).detectRegions(plane)
        for points in regions:
            x, y, w, h = map(int, cv2.boundingRect(points))
            contrast = float(plane[points[:, 1], points[:, 0]].mean())
            raw = dict(box=[x, y, x+w, y+h], color_group=group, mean_contrast=contrast)
            stages["raw"].append(raw)
            if contrast < 15:
                continue
            stages["contrast"].append(raw)
            c = candidate(points, group, rgb.shape)
            if c:
                stages["geometry"].append(c)
    stages["selected"] = select(stages["geometry"])
    if stages["selected"] != experimental(rgb, "mser"):
        raise ValueError("Instrumented search differs from frozen baseline")
    return stages


def explain(target, stages):
    group = {"red_balloon": 0, "blue_balloon": 1}[target["label"]]
    best = {}
    for stage, rows in stages.items():
        c = max((r for r in rows if r["color_group"] == group),
                key=lambda r: iou(target["box"], r["box"]), default=None)
        best[stage] = dict(iou=iou(target["box"], c["box"]) if c else 0., candidate=c)
    reason = "covered"
    for stage, failure in (("raw", "no_localized_mser_region"),
                           ("contrast", "contrast_filtered"),
                           ("geometry", "geometry_filtered"),
                           ("selected", "ranking_or_duplicate_filtered")):
        if best[stage]["iou"] < .5:
            reason = failure
            break
    return dict(**target, reason=reason, stages=best)


def original_frames(manifest, archive_path):
    doc = read_json(manifest)
    if doc.get("training_approved") is not True or sha256(archive_path) != doc["source_archive_sha256"]:
        raise ValueError("Require approved, hash-matched training sources")
    forbidden = {r["source_sha256"] for r in doc["samples"] if r["split"] != "train"}
    grouped = defaultdict(dict)
    for row in doc["samples"]:
        if row["split"] == "train" and row["label"] != "background":
            training_source(row["source_image"])
            if row["source_sha256"] in forbidden:
                raise ValueError("Cross-split source identity")
            grouped[row["source_image"]][row["annotation_id"]] = row
    with zipfile.ZipFile(archive_path) as archive:
        for source, targets in sorted(grouped.items()):
            with Image.open(io.BytesIO(bounded_read(archive, source, 30_000_000))) as im:
                pixels = np.asarray(im.convert("RGB"))
            digest = hashlib.sha256(pixels.tobytes()).hexdigest()
            if any(r["source_sha256"] != digest for r in targets.values()):
                raise ValueError("Source pixels changed")
            truth = [dict(label=r["label"], annotation_id=r["annotation_id"],
                          box=[v*(320/pixels.shape[1] if k % 2 == 0 else 240/pixels.shape[0])
                               for k, v in enumerate(r["box_xyxy"])]) for r in targets.values()]
            yield source, cv2.resize(pixels, (320, 240), interpolation=cv2.INTER_AREA), truth


def indoor_frames(root, review, manifest, archive_path):
    root = Path(root).resolve()
    rows = [r for r in admitted(root, review, manifest) if r["label"] != "background"]
    if sha256(archive_path) != ARCHIVE_SHA:
        raise ValueError("Archive changed")
    with zipfile.ZipFile(archive_path) as archive:
        doc = json.loads(bounded_read(archive, "train/_annotations.coco.json", 50_000_000))
    annotations = {r["id"]: r for r in doc["annotations"]}
    sources = {r["id"]: r for r in doc["images"]}
    for frame in read_json(root / "review.json")["frames"]:
        targets = [r for r in rows if r["frame_id"] == frame["id"]]
        if not targets:
            continue
        training_source(frame["source"])
        path = (root / frame["path"]).resolve()
        if not path.is_relative_to(root) or sha256(path) != frame["sha256"]:
            raise ValueError("Frame escaped queue or changed")
        truth = []
        for r in targets:
            a = annotations[r["annotation_id"]]
            source = sources[a["image_id"]]
            if "train/" + source["file_name"] != r["source_image"]:
                raise ValueError("Annotation identity changed")
            x, y, w, h = a["bbox"]
            truth.append(dict(label=r["label"], annotation_id=r["annotation_id"],
                              box=[x*320/source["width"], y*240/source["height"],
                                   (x+w)*320/source["width"], (y+h)*240/source["height"]]))
        with Image.open(path) as im:
            yield frame["source"], np.asarray(im.convert("RGB")), truth


def audit(frames, sheet_dir, panel):
    outputs, tiles = [], []
    for source, rgb, truth in frames:
        stages = trace(rgb)
        targets = [explain(t, stages) for t in truth]
        outputs.append(dict(source=source, targets=targets,
                            candidate_counts={k: len(v) for k, v in stages.items()}))
        for t in targets:
            if t["reason"] == "covered":
                continue
            tile = Image.new("RGB", (320, 280), "white")
            tile.paste(Image.fromarray(rgb), (0, 40))
            draw = ImageDraw.Draw(tile)
            draw.text((2, 2), f"{panel} id={t['annotation_id']} {t['label']}", fill="black")
            draw.text((2, 16), t["reason"], fill="black")
            for box, color in ((t["box"], "lime"),
                               ((t["stages"]["raw"]["candidate"] or {}).get("box"), "cyan"),
                               ((t["stages"]["selected"]["candidate"] or {}).get("box"), "magenta")):
                if box:
                    draw.rectangle([box[0], box[1]+40, box[2], box[3]+40], outline=color, width=2)
            tiles.append(tile)
    sheet_dir.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(tiles), 9):
        batch = tiles[start:start+9]
        sheet = Image.new("RGB", (960, ((len(batch)+2)//3)*280), "#ddd")
        for i, tile in enumerate(batch):
            sheet.paste(tile, (i % 3*320, i//3*280))
        sheet.save(sheet_dir / f"{panel}-{start//9}.png")
    counts = Counter(f"{t['label']}/{t['reason']}" for f in outputs for t in f["targets"])
    return dict(frames=outputs, counts=dict(sorted(counts.items())))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "original-archive", "queue", "review", "indoor-archive", "sheets", "output"):
        p.add_argument(f"--{name}", required=True, type=Path)
    a = p.parse_args()
    if a.output.exists() or a.sheets.exists():
        raise FileExistsError("Use fresh report and sheet paths")
    result = dict(original=audit(original_frames(a.manifest, a.original_archive), a.sheets, "original"),
                  indoor=audit(indoor_frames(a.queue, a.review, a.manifest, a.indoor_archive), a.sheets, "indoor"))
    write_json(a.output, dict(panels=result, script_sha256=sha256(__file__),
        search_sha256=sha256("scripts/research_balloon_search.py"),
        inputs={name: sha256(path) for name, path in dict(manifest=a.manifest,
            original_archive=a.original_archive, indoor_archive=a.indoor_archive,
            queue=a.queue / "review.json", review=a.review).items()},
        scope="Training-positive candidate-stage attribution; incomplete labels, no false-positive claim",
        classifier_evaluated=False, training_performed=False, test_evaluated=False, deployment_approved=False))
    print({name: report["counts"] for name, report in result.items()})
