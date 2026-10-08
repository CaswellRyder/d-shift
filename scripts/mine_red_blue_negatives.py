"""Quarantined train-only hard negatives; every chosen crop needs visual review."""
import argparse
import hashlib
import io
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from dtr.data import overlap, read_json, sha256, write_json
from dtr.runtime import Predictor
from dtr.vision import RED_BLUE_PROFILE, proposals, validate_model_profile
from scripts.build_public_balloon import ARCHIVE_SHA, INDEX_SHA


def safe_negative(crop_box, balloon_boxes):
    # Generic/ambiguous-color balloons also exclude a candidate from negatives.
    return all(overlap(crop_box, box) == 0 for box in balloon_boxes)


def mine(manifest, model, output, variant="baseline", per_source=2):
    if variant not in ("baseline", "mser") or not 1 <= per_source <= 12:
        raise ValueError("Require baseline/mser and 1..12 candidates per training source")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    archive_path = Path("data/raw/matterport-balloon/balloon_dataset.zip")
    index_path = Path("data/raw/matterport-balloon/review/index.json")
    if sha256(archive_path) != ARCHIVE_SHA or sha256(index_path) != INDEX_SHA:
        raise ValueError("Source checksum changed")
    doc = read_json(manifest)
    sources = {row["source_image"] for row in doc["samples"] if row["split"] == "train"}
    if sources.intersection(row["source_image"] for row in doc["samples"] if row["split"] != "train"):
        raise ValueError("Cross-split source overlap")
    annotations = defaultdict(list)
    for item in read_json(index_path):
        annotations[item["source"]].append(item["box_xyxy"])
    predictor = Predictor(model, allow_unvalidated=True)
    validate_model_profile(predictor.metadata, RED_BLUE_PROFILE)
    output.mkdir(parents=True)
    rows = []
    with zipfile.ZipFile(archive_path) as archive:
        for source in sorted(sources):
            with Image.open(io.BytesIO(archive.read(source))) as opened:
                image = opened.convert("RGB")
            source_hash = hashlib.sha256(image.tobytes()).hexdigest()
            rgb = cv2.resize(np.asarray(image), (320, 240), interpolation=cv2.INTER_AREA)
            truth = [[v*(320/image.width if k % 2 == 0 else 240/image.height)
                      for k, v in enumerate(box)] for box in annotations[source]]
            found = []
            if variant == "baseline":
                candidates = proposals(rgb, "balloon", limit=12, profile=RED_BLUE_PROFILE)
            else:
                from scripts.research_balloon_search import experimental
                candidates = experimental(rgb, variant)
            for c in candidates:
                if not safe_negative(c["crop_box"], truth):
                    continue
                a, b, e, f = c["crop_box"]
                crop = rgb[b:f, a:e]
                prediction = predictor.predict(crop)
                if prediction["accepted"]:
                    found.append((prediction["score"], prediction["label"], c, crop))
            for score, prediction, c, crop in sorted(found, key=lambda x: -x[0])[:per_source]:
                index = len(rows)
                relative = f"crops/{index:03d}.png"
                dest = output / relative
                dest.parent.mkdir(exist_ok=True)
                Image.fromarray(crop).save(dest)
                rows.append(dict(id=index, path=relative, sha256=sha256(dest),
                                 label="background", split="train", session=f"photo:{source_hash}",
                                 source_image=source, source_sha256=source_hash,
                                 crop_box=c["crop_box"], prediction=prediction, score=score))
            chosen = [row for row in rows if row["source_image"] == source]
            if chosen:
                context = image.copy()
                context.thumbnail((1024, 768))
                draw = ImageDraw.Draw(context)
                for row in chosen:
                    box = [v*(context.width/320 if k % 2 == 0 else context.height/240)
                           for k, v in enumerate(row["crop_box"])]
                    draw.rectangle(box, outline="lime", width=2)
                    draw.text((box[0], max(0, box[1]-12)), str(row["id"]), fill="lime")
                (output/"contexts").mkdir(exist_ok=True)
                context.save(output/"contexts"/f"{chosen[0]['id']:03d}.jpg")
    report = dict(training_approved=False, review_required=True, samples=rows,
                  base_manifest_sha256=sha256(manifest), model_sha256=sha256(model),
                  archive_sha256=ARCHIVE_SHA, index_sha256=INDEX_SHA,
                  proposal_variant=variant, per_source_limit=per_source,
                  script_sha256=sha256(__file__),
                  labeling="Zero crop-box overlap with ALL upstream balloon boxes; NOT yet reviewed",
                  split="train only; source disjoint from development validation/test")
    write_json(output / "review.json", report)
    for start in range(0, len(rows), 60):
        sheet = Image.new("RGB", (1000, 6*120), "#ddd")
        draw = ImageDraw.Draw(sheet)
        for i, row in enumerate(rows[start:start+60]):
            x, y = (i % 10)*100, (i // 10)*120
            with Image.open(output/row["path"]) as im:
                sheet.paste(ImageOps.contain(im.convert("RGB"), (96, 88)), (x, y+25))
            draw.text((x, y), f"{row['id']}: {row['prediction'][:1]} {row['score']:.2f}", fill="black")
        sheet.save(output/f"sheet-{start:03d}.jpg")
    print(json.dumps(dict(candidates=len(rows), training_approved=False, output=str(output))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--variant", choices=("baseline", "mser"), default="baseline")
    parser.add_argument("--per-source", type=int, default=2)
    args = parser.parse_args()
    mine(args.manifest, args.model, args.output, args.variant, args.per_source)


if __name__ == "__main__":
    main()
