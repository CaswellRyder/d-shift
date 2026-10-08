"""Train-only bounded proposal ablations; no model, hardware or held-out scores."""
import argparse
import io
import json
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from dtr.data import read_json, sha256
from dtr.tracking import iou
from dtr.vision import RED_BLUE_PROFILE, color_masks, proposals


def candidate(points, group, shape):
    area = cv2.contourArea(cv2.convexHull(points))
    x, y, w, h = map(int, cv2.boundingRect(points))
    height, width = shape[:2]
    if area < 6 or min(w, h) < 4 or max(w/h, h/w) > 3 or w*h > .75*width*height:
        return None
    pad = int(max(w, h)*.12)
    return dict(box=[x, y, x+w, y+h],
                crop_box=[max(0, x-pad), max(0, y-pad), min(width, x+w+pad), min(height, y+h+pad)],
                area=float(area), color_group=group,
                proposal_score=float(np.sqrt(area)*(area/(w*h))**2))


def select(candidates, limit=12):
    queues = [[c for c in sorted(candidates, key=lambda c: -c["proposal_score"])
               if c["color_group"] == group] for group in (0, 1)]
    selected = []
    while any(queues) and len(selected) < limit:
        for queue in queues:
            while queue:
                c = queue.pop(0)
                if not any(iou(c["box"], p["box"]) > .65 for p in selected):
                    selected.append(c)
                    break
            if len(selected) == limit:
                break
    return selected


def experimental(rgb, variant):
    if rgb.shape != (240, 320, 3) or rgb.dtype != np.uint8:
        raise ValueError("Experimental search requires 320x240 RGB uint8")
    if variant == "baseline":
        return proposals(rgb, "balloon", limit=12, profile=RED_BLUE_PROFILE)
    pool = []
    if variant in ("opened", "multisat"):
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        masks = color_masks(rgb, "balloon", RED_BLUE_PROFILE)
        for group, mask in enumerate(masks):
            layers = [mask, cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))]
            if variant == "multisat":
                layers.extend(mask & np.where(hsv[:, :, 1] >= s, 255, 0).astype(np.uint8)
                              for s in (100, 160, 210))
            for layer in layers:
                contours, _ = cv2.findContours(layer, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    c = candidate(contour, group, rgb.shape)
                    if c:
                        pool.append(c)
    elif variant in ("mser", "mser_blend"):
        planes = (cv2.subtract(rgb[:, :, 0], cv2.max(rgb[:, :, 1], rgb[:, :, 2])),
                  cv2.subtract(rgb[:, :, 2], cv2.max(rgb[:, :, 0], rgb[:, :, 1])))
        for group, plane in enumerate(planes):
            regions, _ = cv2.MSER_create(5, 6, 25000).detectRegions(plane)
            for points in regions:
                if float(plane[points[:, 1], points[:, 0]].mean()) < 15:
                    continue
                c = candidate(points, group, rgb.shape)
                if c:
                    pool.append(c)
        if variant == "mser_blend":
            # Same total 12. Baseline gets eight slots, supplements use at most four.
            base = proposals(rgb, "balloon", limit=8, profile=RED_BLUE_PROFILE)
            for c in select(pool, 12):
                if not any(iou(c["box"], p["box"]) > .65 for p in base):
                    base.append(c)
                if len(base) == 12:
                    break
            return base
    else:
        raise ValueError("Unknown experimental search variant")
    return select(pool)


def run(manifest, archive):
    doc = read_json(manifest)
    if sha256(archive) != doc["source_archive_sha256"]:
        raise ValueError("Archive checksum changed")
    images = defaultdict(dict)
    for row in doc["samples"]:
        if row["split"] == "train" and row["label"] != "background":
            images[row["source_image"]][row["annotation_id"]] = row
    variants = {v: dict(rows=[], timing_ms=[]) for v in
                ("baseline", "opened", "multisat", "mser", "mser_blend")}
    with zipfile.ZipFile(archive) as zipped:
        for source, targets in sorted(images.items()):
            with Image.open(io.BytesIO(zipped.read(source))) as opened:
                image = opened.convert("RGB")
                rgb = cv2.resize(np.asarray(image), (320, 240), interpolation=cv2.INTER_AREA)
            for variant, results in variants.items():
                start = time.perf_counter()
                found = experimental(rgb, variant)
                results["timing_ms"].append((time.perf_counter()-start)*1000)
                for row in targets.values():
                    box = [v*(320/image.width if k % 2 == 0 else 240/image.height)
                           for k, v in enumerate(row["box_xyxy"])]
                    group = 0 if row["label"] == "red_balloon" else 1
                    overlap = max((iou(box, c["box"]) for c in found
                                   if c["color_group"] == group), default=0.)
                    results["rows"].append(dict(id=row["annotation_id"], label=row["label"],
                                                best_iou=float(overlap), covered=overlap >= .5))
    for results in variants.values():
        results["counts"] = {label: dict(
            covered=sum(r["covered"] for r in results["rows"] if r["label"] == label),
            total=sum(r["label"] == label for r in results["rows"]))
            for label in ("red_balloon", "blue_balloon")}
        results["desktop_mean_search_ms"] = float(np.mean(results.pop("timing_ms")))
    return dict(scope="train-only search coverage, incomplete scene labels, no classifier",
                manifest_sha256=sha256(manifest), script_sha256=sha256(__file__),
                candidate_limit=12, classification_measured=False, pi_timing_measured=False,
                deployment_approved=False, variants=variants)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = run(args.manifest, "data/raw/matterport-balloon/balloon_dataset.zip")
    args.output.parent.mkdir(exist_ok=True, parents=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: {a: b for a, b in v.items() if a != "rows"}
                      for k, v in report["variants"].items()}, indent=2))


if __name__ == "__main__":
    main()
