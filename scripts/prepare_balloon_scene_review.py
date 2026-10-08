"""Reserve a broader full-scene development family, before any model predictions."""
import argparse
from collections import defaultdict
import io
import json
from pathlib import Path
import re
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageDraw

from dtr.data import read_json, sha256, write_json
from scripts.audit_public_balloon_export import bounded_read, source_group
from scripts.prepare_indoor_balloon_review import ARCHIVE_SHA, MAP, fingerprint, holdout_distance


def development_family(name):
    match = re.fullmatch(r"img_(\d{4})(?:_\d{5})?_jpg\.jpg", source_group(name))
    return f"IMG_{match[1]}" if match else None


def prepare(root, original_manifest, training_queue, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    archive_path = Path(root)/"dataset.zip"
    if sha256(archive_path) != ARCHIVE_SHA:
        raise ValueError("Archive changed")
    base = read_json(original_manifest)
    old_zip = Path("data/raw/matterport-balloon/balloon_dataset.zip")
    if sha256(old_zip) != base["source_archive_sha256"]:
        raise ValueError("Original archive changed")
    hashes = []
    with zipfile.ZipFile(old_zip) as old:
        for source in sorted({r["source_image"] for r in base["samples"] if r["split"] == "train"}):
            with Image.open(io.BytesIO(old.read(source))) as im:
                hashes.append(fingerprint(np.asarray(im.convert("RGB"))))
    queue = read_json(Path(training_queue)/"review.json")
    for row in queue["frames"]:
        path = Path(training_queue)/row["path"]
        if sha256(path) != row["sha256"]:
            raise ValueError("Training frame changed")
        with Image.open(path) as im:
            hashes.append(fingerprint(np.asarray(im.convert("RGB"))))
    output.mkdir(parents=True)
    frames,rejected = [],[]
    with zipfile.ZipFile(archive_path) as archive:
        doc = json.loads(bounded_read(archive,"valid/_annotations.coco.json",50_000_000))
        categories = {r["id"]:r["name"] for r in doc["categories"]}
        by_image = defaultdict(list)
        for a in doc["annotations"]:
            by_image[a["image_id"]].append(a)
        groups = {}
        for row in sorted(doc["images"],key=lambda r:r["file_name"]):
            family = development_family(row["file_name"])
            labels = {categories[a["category_id"]] for a in by_image[row["id"]]}
            if family and labels <= MAP.keys():
                groups.setdefault(family,row)
        eligible = list(groups.values())
        for i in np.linspace(0,len(eligible)-1,min(16,len(eligible)),dtype=int):
            row = eligible[int(i)]
            raw = bounded_read(archive,f"valid/{row['file_name']}",20_000_000)
            with Image.open(io.BytesIO(raw)) as im:
                rgb = np.asarray(im.convert("RGB"))
            distance = holdout_distance(rgb,hashes)
            if distance <= 8:
                rejected.append(dict(source=row["file_name"],nearest_training_hamming=distance))
                continue
            scan = cv2.resize(rgb,(320,240),interpolation=cv2.INTER_AREA)
            path = f"frames/{len(frames):03d}.png"
            (output/"frames").mkdir(exist_ok=True)
            Image.fromarray(scan).save(output/path)
            truth = []
            for a in by_image[row["id"]]:
                x,y,w,h = a["bbox"]
                truth.append(dict(label=MAP[categories[a["category_id"]]],
                    box=[x*320/rgb.shape[1],y*240/rgb.shape[0],(x+w)*320/rgb.shape[1],(y+h)*240/rgb.shape[0]]))
            frames.append(dict(id=len(frames),path=path,sha256=sha256(output/path),source=row["file_name"],
                               source_group=development_family(row["file_name"]),
                               nearest_training_hamming=distance,provisional_truth=truth))
    report = dict(frames=frames,rejected=rejected,archive_sha256=ARCHIVE_SHA,script_sha256=sha256(__file__),
                  base_manifest_sha256=sha256(original_manifest),training_queue_sha256=sha256(Path(training_queue)/"review.json"),
                  evaluation_approved=False,training_approved=False,predictions_run=False,
                  deployment_approved=False,scope="New development family, NOT final test or representative Pi recordings",
                  caveat="Filename families and perceptual screening are not verified recording-session independence")
    write_json(output/"review.json",report)
    for start in range(0,len(frames),8):
        sheet = Image.new("RGB",(1280,530),"#ddd")
        draw = ImageDraw.Draw(sheet)
        for n,row in enumerate(frames[start:start+8]):
            x,y = n%4*320,n//4*265
            with Image.open(output/row["path"]) as im:
                draw_frame = ImageDraw.Draw(im)
                for j,t in enumerate(row["provisional_truth"]):
                    draw_frame.rectangle(t["box"],outline="lime",width=1)
                    draw_frame.text((t["box"][0],t["box"][1]),f"{j}:{t['label'][:1]}",fill="lime")
                sheet.paste(im,(x,y+24))
            draw.text((x,y),f"frame {row['id']} {row['source_group']}",fill="black")
        sheet.save(output/f"frames-{start:03d}.jpg")
    print(dict(frames=len(frames),rejected=len(rejected),queue_sha256=sha256(output/"review.json")))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",required=True)
    p.add_argument("--manifest",required=True)
    p.add_argument("--training-queue",required=True)
    p.add_argument("--output",required=True)
    args = p.parse_args()
    prepare(args.root,args.manifest,args.training_queue,args.output)
