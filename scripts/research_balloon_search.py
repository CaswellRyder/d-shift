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


def remove_contained(candidates):
    """Experimental solid-object assumption, not safe for arbitrary overlapping instances.

    Printed details/highlights inside a balloon are not separate balloon proposals.
    Keep this separate from goal rims and preserve differently colored instances.
    """
    def area(c):
        a, b, e, f = c["box"]
        return (e-a)*(f-b)

    kept = []
    for c in sorted(candidates, key=area, reverse=True):
        a, b, e, f = c["box"]
        contained = False
        for p in kept:
            x, y, z, w = p["box"]
            intersection = max(0, min(e, z)-max(a, x))*max(0, min(f, w)-max(b, y))
            if c["color_group"] == p["color_group"] and intersection >= .95*area(c):
                contained = True
                break
        if not contained:
            kept.append(c)
    return kept


def suppress_confirmed_balloon_parts(observations):
    """Research-only: suppress tiny same-color parts only after a parent is accepted.

    Unlike pre-inference containment filtering, an unverified/background parent
    cannot hide a valid smaller balloon. Same-color overlapping instances can
    still be merged incorrectly; this is not a deployed instance-segmentation rule.
    """
    rows = [dict(row) for row in observations]
    labels = {"red_balloon", "blue_balloon"}
    for child in rows:
        if not child["accepted"] or child["label"] not in labels:
            continue
        a, b, c, d = child["box"]
        area = (c-a)*(d-b)
        if area <= 0:
            continue
        for parent in observations:
            if not parent["accepted"] or parent["label"] != child["label"]:
                continue
            x, y, z, w = parent["box"]
            parent_area = (z-x)*(w-y)
            overlap = max(0, min(c, z)-max(a, x))*max(0, min(d, w)-max(b, y))
            if area <= .35*parent_area and overlap >= .95*area:
                child.update(accepted=False, suppressed=True, suppression_reason="confirmed_balloon_part")
                break
    return rows


def split_merged_balloon_parents(observations, min_fraction=.15, max_fraction=.6, max_aspect=1.5):
    """Research-only: reject an accepted box that merges two separate same-color balloons.

    Apply after duplicate suppression and before confirmed-part suppression. A
    parent is rejected when it contains two accepted, roughly round, same-color
    children that do not overlap each other; the children are kept. A single
    balloon split into two halves is guarded against only by the roundness and
    size limits, so this is a geometric heuristic, not instance segmentation.
    """
    rows = [dict(row) for row in observations]
    labels = {"red_balloon", "blue_balloon"}

    def area(box):
        return max(0, box[2]-box[0])*max(0, box[3]-box[1])

    def round_child(child, parent):
        a, p = child["box"], parent["box"]
        w, h = a[2]-a[0], a[3]-a[1]
        overlap = area([max(a[0], p[0]), max(a[1], p[1]), min(a[2], p[2]), min(a[3], p[3])])
        return (child["accepted"] and child["label"] == parent["label"]
                and w > 0 and h > 0 and max(w/h, h/w) <= max_aspect
                and min_fraction*area(p) <= area(a) <= max_fraction*area(p)
                and overlap >= .9*area(a))

    merged = []
    for i, parent in enumerate(observations):
        if not parent["accepted"] or parent["label"] not in labels:
            continue
        children = [c for j, c in enumerate(observations) if j != i and round_child(c, parent)]
        if any(iou(a["box"], b["box"]) < .1 for k, a in enumerate(children) for b in children[k+1:]):
            merged.append(i)
    for i in merged:
        rows[i].update(accepted=False, suppressed=True, suppression_reason="merged_balloon_parent")
    return rows


def replace_oversized_balloon_parents(observations, min_fraction=.15, max_fraction=.35):
    """Research-only: replace an accepted box by its largest substantial same-color part.

    Apply after merged-parent splitting and before confirmed-part suppression. A
    correct balloon box contains only tiny accepted parts (highlights, prints); a
    box spanning a balloon and nearby clutter contains the balloon as a part of
    15-35% of its area. Largest parents first: the parent is rejected, its largest
    such part is kept, and its other parts stay suppressed as they would have been.
    Designed on development panels 1-2 only; a geometric heuristic, not segmentation.
    """
    rows = [dict(row) for row in observations]
    labels = {"red_balloon", "blue_balloon"}

    def area(box):
        return max(0, box[2]-box[0])*max(0, box[3]-box[1])

    def part(child, parent):
        a, p = child["box"], parent["box"]
        overlap = area([max(a[0], p[0]), max(a[1], p[1]), min(a[2], p[2]), min(a[3], p[3])])
        return (child is not parent and child["accepted"] and child["label"] == parent["label"]
                and area(a) <= max_fraction*area(p) and overlap >= .95*area(a))

    kept = set()
    for parent in sorted(rows, key=lambda r: -area(r["box"])):
        if id(parent) in kept or not parent["accepted"] or parent["label"] not in labels:
            continue
        parts = [c for c in rows if part(c, parent)]
        large = [c for c in parts if area(c["box"]) >= min_fraction*area(parent["box"])]
        if not large:
            continue
        keep = max(large, key=lambda c: area(c["box"]))
        kept.add(id(keep))
        parent.update(accepted=False, suppressed=True, suppression_reason="oversized_balloon_parent")
        for c in parts:
            if c is not keep:
                c.update(accepted=False, suppressed=True, suppression_reason="confirmed_balloon_part")
    return rows


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


def blue_level_candidates(rgb):
    """Fixed research supplement for regions that are not stable MSER extrema.

    Unchanged opponent-blue contrast, no learned thresholds. Raw and opened
    masks retain small targets and offer a second view across narrow bridges.
    This is not a balloon decision; every returned region still needs inference.
    """
    plane = cv2.subtract(rgb[:, :, 2], cv2.max(rgb[:, :, 0], rgb[:, :, 1]))
    pool = []
    kernel = np.ones((3, 3), np.uint8)
    for threshold in (16, 32, 64, 96, 128, 160, 192, 224):
        mask = cv2.compare(plane, threshold, cv2.CMP_GE)
        for layer in (mask, cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)):
            contours, _ = cv2.findContours(layer, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                c = candidate(contour, 1, rgb.shape)
                if c:
                    pool.append(c)
    return select(pool)


def instance_split_candidates(rgb):
    """Research-only watershed of multi-lobed opponent-color components.

    Fill interior highlight holes before distance seeding. A component must have
    2..4 disconnected thick cores to be split; a single core adds no candidates.
    Fixed geometry, not learned or tuned on development/qualification frames.
    """
    pool = []
    planes = (cv2.subtract(rgb[:, :, 0], cv2.max(rgb[:, :, 1], rgb[:, :, 2])),
              cv2.subtract(rgb[:, :, 2], cv2.max(rgb[:, :, 0], rgb[:, :, 1])))
    for group, plane in enumerate(planes):
        contours, _ = cv2.findContours(cv2.compare(plane, 15, cv2.CMP_GE),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Bound work in highly textured scenes independently of object labels.
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:16]:
            x, y, w, h = cv2.boundingRect(contour)
            if cv2.contourArea(contour) < 40 or min(w, h) < 8:
                continue
            mask = np.zeros((h+2, w+2), np.uint8)
            cv2.drawContours(mask, [contour - np.array([[[x-1, y-1]]])], -1, 255, -1)
            distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
            if distance.max() < 4:
                continue
            count, cores = cv2.connectedComponents((distance >= .6*distance.max()).astype(np.uint8))
            if not 2 <= count-1 <= 4:
                continue
            markers = (cores + 1).astype(np.int32)
            markers[(mask != 0) & (cores == 0)] = 0
            # Image gradients guide growth; the filled mask fixes the background.
            padded = np.zeros((h+2, w+2, 3), np.uint8)
            padded[1:-1, 1:-1] = rgb[y:y+h, x:x+w]
            cv2.watershed(padded, markers)
            for label in range(2, count+1):
                ys, xs = np.nonzero((markers == label) & (mask != 0))
                if len(xs) < .12*np.count_nonzero(mask):
                    continue
                points = np.column_stack((xs+x-1, ys+y-1)).astype(np.int32)
                c = candidate(points, group, rgb.shape)
                if c:
                    pool.append(c)
    return select(pool)


def experimental(rgb, variant):
    if rgb.shape != (240, 320, 3) or rgb.dtype != np.uint8:
        raise ValueError("Experimental search requires 320x240 RGB uint8")
    if variant == "baseline":
        return proposals(rgb, "balloon", limit=12, profile=RED_BLUE_PROFILE)
    if variant in ("mser_blue_levels", "mser_instance_split"):
        # Preserve every baseline slot; at most four additions, twelve total.
        base = experimental(rgb, "mser")
        additions = 0
        supplement = blue_level_candidates if variant == "mser_blue_levels" else instance_split_candidates
        for c in supplement(rgb):
            if len(base) >= 12 or additions >= 4:
                break
            if not any(iou(c["box"], p["box"]) > .65 for p in base):
                base.append(c)
                additions += 1
        return base
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
    elif variant in ("mser", "mser_blend", "mser_components", "mser_confirmed_parts", "mser_chromatic", "mser_blue_red"):
        planes = (cv2.subtract(rgb[:, :, 0], cv2.max(rgb[:, :, 1], rgb[:, :, 2])),
                  cv2.subtract(rgb[:, :, 2], cv2.max(rgb[:, :, 0], rgb[:, :, 1])))
        if variant == "mser_blue_red":
            # Research-only: retain cyan/blue contrast when green approaches blue.
            # Gray stays zero, but cyan clutter can also become a proposal.
            planes = (planes[0], cv2.subtract(rgb[:, :, 2], rgb[:, :, 0]))
        if variant == "mser_chromatic":
            # Normalize opponent-color contrast by brightness. Research-only:
            # dark colored regions gain contrast, but noise can gain it too.
            total = rgb.astype(np.float32).sum(axis=2)
            valid = rgb.max(axis=2) >= 30
            planes = tuple(np.where(valid, np.clip(p.astype(np.float32)*255/
                           np.maximum(total, 1), 0, 255), 0).astype(np.uint8) for p in planes)
        for group, plane in enumerate(planes):
            regions, _ = cv2.MSER_create(5, 6, 25000).detectRegions(plane)
            for points in regions:
                if float(plane[points[:, 1], points[:, 0]].mean()) < 15:
                    continue
                c = candidate(points, group, rgb.shape)
                if c:
                    pool.append(c)
        if variant == "mser_components":
            pool = remove_contained(pool)
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
            if Path(row["source_image"]).name.lower().startswith("img_"):
                raise ValueError("Reserved IMG family cannot enter training search audit")
            images[row["source_image"]][row["annotation_id"]] = row
    variants = {v: dict(rows=[], timing_ms=[]) for v in
                ("baseline", "opened", "multisat", "mser", "mser_blend", "mser_components", "mser_chromatic", "mser_blue_red", "mser_blue_levels", "mser_instance_split")}
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
