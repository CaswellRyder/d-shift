"""Quarantine a bounded indoor training subset with source-family and holdout checks."""
import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from dtr.data import read_json, sha256, write_json
from dtr.vision import RED_BLUE_PROFILE, proposals
from scripts.audit_public_balloon_export import bounded_read, source_group
from scripts.mine_red_blue_negatives import safe_negative

ARCHIVE_SHA = "da21324671a143c3783685443e913a6a611c5b3aaa5249231b34567bbdf84c5e"
MAP = {"red_ballon": "red_balloon", "blue_ballon": "blue_balloon"}


def family(filename):
    match = re.fullmatch(r"frame_(\d{4}|\d{6})_jpg\.jpg", source_group(filename))
    return f"indoor-frame-{len(match[1])}-digits" if match else None


def fingerprint(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    values = cv2.dct(gray)[:8, :8].reshape(-1)[1:]
    return values > np.median(values)


def holdout_distance(rgb, fingerprints):
    return min(int(np.count_nonzero(fingerprint(view) != h))
               for view in (rgb, rgb[::-1], rgb[:, ::-1], rgb[::-1, ::-1]) for h in fingerprints)


def build(root, manifest, output, per_family=8, exclude_queue=None):
    if type(per_family) is not int or not 1 <= per_family <= 24:
        raise ValueError("Require 1..24 candidate frames per training family")
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError(output)
    archive_path = root / "dataset.zip"
    if sha256(archive_path) != ARCHIVE_SHA:
        raise ValueError("Public source archive changed")
    excluded_groups, prior_fingerprints = set(), []
    if exclude_queue:
        prior_path = Path(exclude_queue).resolve()
        prior = read_json(prior_path)
        if prior["archive_sha256"] != ARCHIVE_SHA or prior["base_manifest_sha256"] != sha256(manifest):
            raise ValueError("Prior queue identity changed")
        for frame in prior["frames"]:
            path = (prior_path.parent / frame["path"]).resolve()
            if (not path.is_relative_to(prior_path.parent) or sha256(path) != frame["sha256"]
                    or family(Path(frame["source"]).name) is None):
                raise ValueError("Invalid prior training frame")
            excluded_groups.add(source_group(Path(frame["source"]).name))
            with Image.open(path) as im:
                prior_fingerprints.append(fingerprint(np.asarray(im.convert("RGB"))))
    base = read_json(manifest)
    old_archive = Path("data/raw/matterport-balloon/balloon_dataset.zip")
    if sha256(old_archive) != base["source_archive_sha256"]:
        raise ValueError("Original source archive changed")
    heldout_sources = {r["source_image"] for r in base["samples"] if r["split"] != "train"}
    fingerprints = []
    with zipfile.ZipFile(old_archive) as old:
        for source in sorted(heldout_sources):
            # Automated duplicate check only: no display, predictions, labels or fitting.
            with Image.open(io.BytesIO(old.read(source))) as im:
                fingerprints.append(fingerprint(np.asarray(im.convert("RGB"))))
    output.mkdir(parents=True)
    rows, frames, rejected, seen = [], [], [], set()
    with zipfile.ZipFile(archive_path) as archive:
        splits = {s: json.loads(bounded_read(archive, f"{s}/_annotations.coco.json", 50_000_000))
                  for s in ("train", "valid", "test")}
        forbidden = {source_group(r["file_name"]) for s in ("valid", "test") for r in splits[s]["images"]}
        train = splits["train"]
        categories = {r["id"]: r["name"] for r in train["categories"]}
        anns = defaultdict(list)
        for a in train["annotations"]:
            anns[a["image_id"]].append(a)
        choices = defaultdict(dict)
        for row in sorted(train["images"], key=lambda r:r["file_name"]):
            f = family(row["file_name"])
            group = source_group(row["file_name"])
            labels = {categories[a["category_id"]] for a in anns[row["id"]]}
            if f and group not in forbidden | excluded_groups and labels and labels <= MAP.keys():
                choices[f].setdefault(group, row)
        for f, groups in sorted(choices.items()):
            eligible = sorted(groups.values(), key=lambda r:r["file_name"])
            for idx in np.linspace(0, len(eligible)-1, min(per_family, len(eligible)), dtype=int):
                row = eligible[int(idx)]
                raw = bounded_read(archive, f"train/{row['file_name']}", 20_000_000)
                with Image.open(io.BytesIO(raw)) as im:
                    rgb = np.asarray(im.convert("RGB"))
                digest = hashlib.sha256(rgb.tobytes()).hexdigest()
                distance = holdout_distance(rgb, fingerprints)
                prior_distance = holdout_distance(rgb, prior_fingerprints) if prior_fingerprints else None
                if distance <= 8 or digest in seen or (prior_distance is not None and prior_distance <= 8):
                    rejected.append(dict(source=row["file_name"], holdout_hamming=distance,
                                         prior_training_hamming=prior_distance, duplicate=digest in seen))
                    continue
                seen.add(digest)
                scan = cv2.resize(rgb, (320, 240), interpolation=cv2.INTER_AREA)
                frame_id = len(frames)
                full = f"frames/{frame_id:03d}.png"
                (output / "frames").mkdir(exist_ok=True)
                Image.fromarray(scan).save(output / full)
                boxes = []
                for a in anns[row["id"]]:
                    x, y, w, h = a["bbox"]
                    boxes.append([x*320/rgb.shape[1], y*240/rgb.shape[0],
                                  (x+w)*320/rgb.shape[1], (y+h)*240/rgb.shape[0]])
                candidates = []
                for annotation, box in zip(anns[row["id"]], boxes):
                    a,b,c,d = [int(round(v)) for v in box]
                    if min(c-a,d-b) < 6:
                        continue
                    pad = int(max(c-a,d-b)*.12)
                    candidates.append(dict(crop_box=[max(0,a-pad),max(0,b-pad),min(320,c+pad),min(240,d+pad)],
                                           label=MAP[categories[annotation["category_id"]]],
                                           annotation_id=annotation["id"]))
                # Missing boxes are NOT background approval; these still need visual review.
                # Expansion reviews use the same proposal shape as the research detector.
                if exclude_queue:
                    from scripts.research_balloon_search import experimental
                    proposed = experimental(scan, "mser")
                else:
                    proposed = proposals(scan,"balloon",limit=12,profile=RED_BLUE_PROFILE)
                negatives = [c for c in proposed
                             if safe_negative(c["crop_box"], boxes)]
                candidates.extend(dict(crop_box=c["crop_box"],label="background",annotation_id=None)
                                  for c in negatives[:3])
                ids = []
                for c in candidates:
                    a,b,z,w = c["crop_box"]
                    relative = f"crops/{len(rows):03d}.png"
                    (output / "crops").mkdir(exist_ok=True)
                    Image.fromarray(scan[b:w,a:z]).save(output / relative)
                    ids.append(len(rows))
                    rows.append(dict(id=len(rows),path=relative,sha256=sha256(output/relative),
                                     split="train",session=f"engdes2:{f}",source_image=f"train/{row['file_name']}",
                                     source_sha256=digest,source_group=source_group(row["file_name"]),
                                     frame_id=frame_id,**c))
                frames.append(dict(id=frame_id,path=full,sha256=sha256(output/full),family=f,
                                   source=row["file_name"],holdout_hamming=distance,
                                   prior_training_hamming=prior_distance,crop_ids=ids,boxes=boxes))
    queue = dict(archive_sha256=ARCHIVE_SHA,base_manifest_sha256=sha256(manifest),samples=rows,frames=frames,
                 rejected=rejected,script_sha256=sha256(__file__),training_approved=False,review_required=True,
                 deployment_approved=False,test_evaluated=False,
                 holdout_pixel_use="Automated perceptual duplicate screening only; no display/inference/fitting",
                 caveat="Two filename families assigned training-only; no independent new benchmark. Perceptual hash is heuristic.",
                 class_mapping=MAP, candidate_frames_per_family=per_family,
                 excluded_queue_sha256=sha256(exclude_queue) if exclude_queue else None,
                 negative_search="mser" if exclude_queue else "baseline",
                 search_sha256=sha256("scripts/research_balloon_search.py") if exclude_queue else None)
    write_json(output / "review.json", queue)
    for start in range(0,len(rows),48):
        sheet = Image.new("RGB",(960,720),"#ddd")
        draw = ImageDraw.Draw(sheet)
        for n,r in enumerate(rows[start:start+48]):
            x,y = n%8*120,n//8*120
            with Image.open(output/r["path"]) as im:
                sheet.paste(ImageOps.contain(im,(116,92)),(x,y+24))
            draw.text((x,y),f"{r['id']} {r['label']}",fill="black")
        sheet.save(output/f"crops-{start:03d}.jpg")
    for start in range(0,len(frames),8):
        sheet = Image.new("RGB",(1280,530),"#ddd")
        draw = ImageDraw.Draw(sheet)
        for n,r in enumerate(frames[start:start+8]):
            x,y = n%4*320,n//4*265
            with Image.open(output/r["path"]) as im:
                sheet.paste(im,(x,y+24))
            draw.text((x,y),f"frame {r['id']} crops {r['crop_ids']}",fill="black")
        sheet.save(output/f"frames-{start:03d}.jpg")
    print(dict(frames=len(frames),crops=len(rows),counts=dict(Counter(r["label"] for r in rows)),
               rejected=len(rejected),queue_sha256=sha256(output/"review.json")))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",required=True)
    p.add_argument("--manifest",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--per-family",type=int,default=8)
    p.add_argument("--exclude-queue",help="Prior training review.json; exclude its groups and near-duplicate frames")
    args = p.parse_args()
    build(args.root,args.manifest,args.output,args.per_family,args.exclude_queue)
