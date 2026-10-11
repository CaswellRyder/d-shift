"""Quarantine new public TRAIN frames with cross-source duplicate screening.

The export's FRC class name is NOT automatically mapped to balloons. Every crop
remains unreviewed. Held-out pixels are used only for automated duplicate checks,
never displayed, classified or used for fitting by this script.
"""
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
from dtr.tracking import iou
from scripts.audit_public_balloon_export import bounded_read, source_group
from scripts.prepare_indoor_balloon_review import ARCHIVE_SHA as ENGDES_ARCHIVE_SHA
from scripts.prepare_indoor_balloon_review import fingerprint, holdout_distance
from scripts.research_balloon_search import experimental

ARCHIVE_SHA = "183eea2ff198efe8833a4c110c1d4a890dd51562f10818175685e1c114e7e305"
ASTEK_ARCHIVE_SHA = "693b73d058ff79ec45c567a384589db945068490d16b7de9bf27b20b41ba2499"
REFERENCE_QUEUES = (
    "engdes2-indoor-review-20261008", "engdes2-indoor-expansion-review-20261008",
    "engdes2-indoor-expansion2-review-20261011", "engdes2-development-scenes-20261008",
    "engdes2-development-panel2-20261010", "engdes2-lockbox-panel3-20261010",
)


def family(filename):
    name = source_group(filename)
    if name.startswith("img_"):
        raise ValueError("Reserved IMG-like source cannot enter training quarantine")
    if re.match(r"left\d+", name):
        return "left-sequence"
    if re.match(r"blue[-_]", name):
        return "blue-sequence"
    return "other-unverified"


def astek_family(filename):
    # This separate pinned archive has its own IMG_ names. It is not EngDes2;
    # all EngDes2 IMG_ pixels remain exclusion references, never training input.
    match = re.fullmatch(r"img_(20250714|20250716)_\d{6}(?:_\d+)?_jpg\.jpg", source_group(filename))
    return "astek-date-"+match[1] if match else None


def decoded(archive, member, row):
    with Image.open(io.BytesIO(bounded_read(archive, member, 20_000_000))) as im:
        if im.width*im.height > 16_000_000 or im.size != (row["width"],row["height"]):
            raise ValueError("Source dimensions changed or exceed bound")
        return np.asarray(im.convert("RGB"))


def scan_box(bbox, width, height):
    if len(bbox) != 4 or not np.isfinite(bbox).all():
        raise ValueError("Invalid source box")
    x,y,w,h = bbox
    if not (width > 0 and height > 0 and w > 0 and h > 0
            and 0 <= x < x+w <= width and 0 <= y < y+h <= height):
        raise ValueError("Invalid source box bounds")
    return [x*320/width,y*240/height,(x+w)*320/width,(y+h)*240/height]


def crop_box(box):
    a,b,c,d = box
    pad = int(max(c-a,d-b)*.12)
    return [max(0,int(np.floor(a))-pad),max(0,int(np.floor(b))-pad),
            min(320,int(np.ceil(c))+pad),min(240,int(np.ceil(d))+pad)]


def references():
    prints, receipts = [], {}
    manifest = Path("data/balloon-red-blue-bootstrap-20261008/manifest.json")
    doc = read_json(manifest)
    archive_path = Path("data/raw/matterport-balloon/balloon_dataset.zip")
    if sha256(archive_path) != doc["source_archive_sha256"]:
        raise ValueError("Reference archive changed")
    receipts[str(manifest)] = sha256(manifest)
    receipts[str(archive_path)] = sha256(archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        for source in sorted({r["source_image"] for r in doc["samples"]}):
            with Image.open(io.BytesIO(bounded_read(archive, source, 30_000_000))) as im:
                rgb = np.asarray(im.convert("RGB"))
            expected = {r["source_sha256"] for r in doc["samples"] if r["source_image"] == source}
            if expected != {hashlib.sha256(rgb.tobytes()).hexdigest()}:
                raise ValueError("Reference source pixels changed")
            prints.append(fingerprint(rgb))
    for name in REFERENCE_QUEUES:
        root, review_path = Path("data")/name, Path("configs")/(name+".json")
        queue_path = root/"review.json"
        if read_json(review_path)["queue_sha256"] != sha256(queue_path):
            raise ValueError("Reference queue changed")
        receipts[str(queue_path)], receipts[str(review_path)] = sha256(queue_path),sha256(review_path)
        for row in read_json(queue_path)["frames"]:
            path = (root/row["path"]).resolve()
            if not path.is_relative_to(root.resolve()) or sha256(path) != row["sha256"]:
                raise ValueError("Reference frame changed or escaped root")
            with Image.open(path) as im:
                prints.append(fingerprint(np.asarray(im.convert("RGB"))))
    # Screen against the whole reserved IMG family, not only previously reviewed
    # development panels. These pixels are never displayed or classified here.
    engdes_path = Path("data/raw/engdes2-red-blue-v1-20261008/dataset.zip")
    if sha256(engdes_path) != ENGDES_ARCHIVE_SHA:
        raise ValueError("Reserved-family archive changed")
    receipts[str(engdes_path)] = sha256(engdes_path)
    with zipfile.ZipFile(engdes_path) as archive:
        for split in ("train","valid","test"):
            doc = json.loads(bounded_read(archive,f"{split}/_annotations.coco.json",50_000_000))
            for row in doc["images"]:
                if row["file_name"].lower().startswith("img_"):
                    prints.append(fingerprint(decoded(archive,split+"/"+row["file_name"],row)))
    return prints,receipts


def prepare(root, output, per_family=8, source_profile="frc-review"):
    if source_profile not in ("frc-review", "astek-balloons"):
        raise ValueError("Unknown pinned source profile")
    astek = source_profile == "astek-balloons"
    expected_archive = ASTEK_ARCHIVE_SHA if astek else ARCHIVE_SHA
    if type(per_family) is not int or not 1 <= per_family <= 12:
        raise ValueError("Require 1..12 frames per filename family")
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError(output)
    archive_path = root/"dataset.zip"
    receipt = read_json(root/"download-receipt.json")
    if sha256(archive_path) != expected_archive or receipt["archive_sha256"] != expected_archive:
        raise ValueError("Public source archive changed")
    reference, reference_receipts = references()
    output.mkdir(parents=True)
    for name in ("frames","native","crops"):
        (output/name).mkdir()
    rows,frames,rejected = [],[],[]
    counts = Counter()
    selected_prints = []
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive members")
        split_names = ("train",) if astek else ("train","valid")
        actual_splits = {n.split("/")[0] for n in names if n.endswith("/_annotations.coco.json")}
        if actual_splits != set(split_names):
            raise ValueError("Pinned source split layout changed")
        splits = {s:json.loads(bounded_read(archive,f"{s}/_annotations.coco.json",50_000_000))
                  for s in split_names}
        train = splits["train"]
        held_rows = splits.get("valid",{}).get("images",[])
        forbidden = {source_group(r["file_name"]) for r in held_rows}
        holdout = [fingerprint(decoded(archive,"valid/"+r["file_name"],r)) for r in held_rows]
        annotations = defaultdict(list)
        categories = {r["id"]:r["name"] for r in train["categories"]}
        for a in train["annotations"]:
            annotations[a["image_id"]].append(a)
        unique = {}
        for image in sorted(train["images"],key=lambda r:r["file_name"]):
            unique.setdefault(source_group(image["file_name"]),image)
        choices = sorted(unique.items(),key=lambda item:hashlib.sha256(item[0].encode()).hexdigest())
        for group, image in choices:
            if group in forbidden or (not astek and group.startswith("img_")):
                rejected.append(dict(source_group=group,reason="heldout_or_reserved_source"))
                continue
            fam = astek_family(image["file_name"]) if astek else family(image["file_name"])
            if fam is None:
                continue
            if astek and not any(categories[a["category_id"]] in ("red balloon","blue balloon")
                                 for a in annotations[image["id"]]):
                continue
            if counts[fam] >= per_family:
                continue
            rgb = decoded(archive,"train/"+image["file_name"],image)
            ref_distance = holdout_distance(rgb,reference)
            held_distance = holdout_distance(rgb,holdout) if holdout else None
            selected_distance = holdout_distance(rgb,selected_prints) if selected_prints else None
            if (ref_distance <= 8 or (held_distance is not None and held_distance <= 8)
                    or (selected_distance is not None and selected_distance <= 8)):
                rejected.append(dict(source_group=group,reason="perceptual_duplicate",reference_hamming=ref_distance,
                                     upstream_holdout_hamming=held_distance,selected_hamming=selected_distance))
                continue
            truth = [dict(annotation_id=a["id"],source_label=categories[a["category_id"]],
                          box=scan_box(a["bbox"],image["width"],image["height"]))
                     for a in annotations[image["id"]]]
            if not truth:
                continue
            scan = cv2.resize(rgb,(320,240),interpolation=cv2.INTER_AREA)
            hypotheses = [dict(t,kind="upstream_object",crop_box=crop_box(t["box"])) for t in truth
                          if (not astek or t["source_label"] in ("red balloon","blue balloon"))
                          if min(t["box"][2]-t["box"][0],t["box"][3]-t["box"][1]) >= 4]
            candidates = [c for c in experimental(scan,"mser") if max((iou(c["box"],t["box"]) for t in truth),default=0) < .05]
            hypotheses += [dict(c,kind="unmatched_proposal") for c in candidates[:2]]
            if not hypotheses:
                continue
            index = len(frames)
            path,native = f"frames/{index:03d}.png",f"native/{index:03d}.png"
            Image.fromarray(scan).save(output/path)
            Image.fromarray(rgb).save(output/native)
            frames.append(dict(id=index,path=path,sha256=sha256(output/path),native_path=native,
                native_sha256=sha256(output/native),source="train/"+image["file_name"],source_group=group,
                filename_family=fam,source_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),truth=truth,
                reference_hamming=ref_distance,upstream_holdout_hamming=held_distance,selected_hamming=selected_distance))
            for hypothesis in hypotheses:
                a,b,c,d = hypothesis["crop_box"]
                relative = f"crops/{len(rows):03d}.png"
                Image.fromarray(scan[b:d,a:c]).save(output/relative)
                rows.append(dict(hypothesis,id=len(rows),path=relative,sha256=sha256(output/relative),frame_id=index,
                                 split="train",label="unreviewed",source_image=frames[-1]["source"],source_group=group))
            counts[fam] += 1
            selected_prints.append(fingerprint(rgb))
    doc = dict(frames=frames,samples=rows,rejected=rejected,filename_family_counts=dict(counts),
        archive_sha256=expected_archive,source_profile=source_profile,
        download_receipt_sha256=sha256(root/"download-receipt.json"),
        reference_inputs=reference_receipts,script_sha256=sha256(__file__),
        source=receipt["source"],training_approved=False,deployment_approved=False,review_required=True,
        label_mapping_approved=False,heldout_pixels_used_for_duplicate_screening_only=True,
        test_evaluated=False,independent_recording_sessions_verified=False,
        warning=("Upstream source is train-only, not a new held-out test. Date groups are filename hints, not verified independent sessions. All crops need visual color and object review; unmatched is not automatically background."
                 if astek else "Upstream class name is FRC 2022 - v1 ball. Inspect object identity; never assume balloon or turn an unmatched proposal into background."))
    write_json(output/"review.json",doc)
    for kind,items,columns,tile in (("crops",rows,6,(160,160)),("frames",frames,3,(320,270))):
        per_sheet = columns*4
        for start in range(0,len(items),per_sheet):
            sheet = Image.new("RGB",(columns*tile[0],4*tile[1]),"#ddd")
            draw = ImageDraw.Draw(sheet)
            for i,r in enumerate(items[start:start+per_sheet]):
                x,y = i%columns*tile[0],i//columns*tile[1]
                with Image.open(output/r["path"]) as im:
                    picture = im.convert("RGB")
                if kind == "frames":
                    mark = ImageDraw.Draw(picture)
                    for t in r["truth"]:
                        mark.rectangle(t["box"],outline="lime",width=1)
                sheet.paste(ImageOps.contain(picture,(tile[0]-4,tile[1]-25)),(x,y+25))
                draw.text((x+2,y+2),f"{kind} {r['id']}"+(f" / frame {r['frame_id']}" if kind=="crops" else ""),fill="black")
            sheet.save(output/f"{kind}-{start:03d}.jpg")
    return dict(frames=len(frames),crops=len(rows),families=dict(counts),rejected=len(rejected),queue_sha256=sha256(output/"review.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--per-family",type=int,default=8)
    parser.add_argument("--source-profile",choices=("frc-review","astek-balloons"),default="frc-review")
    args = parser.parse_args()
    print(prepare(args.root,args.output,args.per_family,args.source_profile))
