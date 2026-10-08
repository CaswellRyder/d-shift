"""Build a checksum-pinned, AI-reviewed REAL PHOTO bootstrap; not a DTR benchmark.

Run review_balloon_source.py first. All crops from one source image stay together.
No synthetic recoloring or auto-generated positive labels are used.
"""

import argparse
import hashlib
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image

from dtr.data import read_json, sha256, validate, write_json

ARCHIVE_SHA = "053797b8a0f3c210b80d1494af99e2e65c7830c5dccca581285e61048cb0bf1c"
INDEX_SHA = "a66b696b6bdc9f4900151cc4970ffcad1dcf03e2e4749d90236ad77322751ba8"
# Source-image holdouts chosen before training, not from model results.
HOLDOUT_ANCHORS = {"val": [129, 159, 193], "test": [199, 214, 275]}


def build(archive_path, review_path, index_path, output, config, *, training_only=False):
    if sha256(archive_path) != ARCHIVE_SHA:
        raise ValueError("Unexpected source archive checksum")
    if sha256(index_path) != INDEX_SHA:
        raise ValueError("Review index changed; color IDs must be re-reviewed")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    review, index = read_json(review_path), read_json(index_path)
    by_id = {row["id"]: row for row in index}
    holdouts = ({} if training_only else
                {by_id[i]["source"]: split for split, ids in HOLDOUT_ANCHORS.items() for i in ids})
    selected = [(i, label) for label in config["classes"] for i in review[label]]
    if len({i for i, _ in selected}) != len(selected):
        raise ValueError("Duplicate reviewed crop ID")
    if training_only and any(not by_id[i]["source"].startswith("balloon/train/")
                             for i, _ in selected):
        raise ValueError("Training-only bootstrap cannot consume upstream validation images")
    rows, source_splits = [], {}
    output.mkdir(parents=True)
    with zipfile.ZipFile(archive_path) as archive:
        for i, label in selected:
            item = by_id[i]
            source = item["source"]
            with Image.open(io.BytesIO(archive.read(source))) as opened:
                image = opened.convert("RGB")
            digest = hashlib.sha256(image.tobytes()).hexdigest()
            split = holdouts.get(source, "train")
            if digest in source_splits and source_splits[digest] != split:
                raise ValueError("Duplicate full source image across splits")
            source_splits[digest] = split
            crop = image.crop(item["box_xyxy"])
            if min(crop.size) < 12:
                continue
            relative = f"images/{split}/{i:03d}.png"
            dest = output / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            crop.save(dest)
            rows.append(
                dict(
                    path=relative,
                    split=split,
                    label=label,
                    session=f"photo:{digest}",
                    sha256=sha256(dest),
                    source_image=source,
                    source_sha256=digest,
                    annotation_id=i,
                    box_xyxy=item["box_xyxy"],
                    label_origin="AI visual color review of upstream balloon polygon",
                )
            )
    qualification = {
        "domain": "generic_real_balloon_photographs_NOT_DTR",
        "label_review": "AI reviewed; not human-certified",
        "recording_session_independence": "UNVERIFIED; source-image disjoint only",
        "negative_coverage": "other-color balloons only; arena hard negatives still needed",
        "competition_accuracy_established": False,
        "redistribution_approved": False,
        "training_only": training_only,
    }
    manifest = output / "manifest.json"
    write_json(
        manifest,
        dict(
            task=config["task"],
            classes=config["classes"],
            synthetic=False,
            source=review["source_url"],
            source_archive_sha256=ARCHIVE_SHA,
            review_sha256=sha256(review_path),
            qualification=qualification,
            split_provenance=("upstream train images only; no validation/test split" if training_only
                              else "source-image holdouts; session independence UNVERIFIED"),
            samples=rows,
        ),
    )
    validate(manifest, config, splits=("train",) if training_only else ("train", "val", "test"))
    counts = Counter(f"{r['split']}/{r['label']}" for r in rows)
    receipt = dict(
        samples=len(rows),
        distinct_source_images=len(source_splits),
        counts=dict(sorted(counts.items())),
        manifest_sha256=sha256(manifest),
        qualification=qualification,
        source_url=review["source_url"],
        archive_sha256=ARCHIVE_SHA,
    )
    write_json(output / "receipt.json", receipt)
    write_json(output / "color-review.json", review)
    print(json.dumps(receipt, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/balloon-public-bootstrap-20261003")
    parser.add_argument("--config", default="configs/balloon.json")
    parser.add_argument("--review", default="configs/balloon-public-review.json")
    parser.add_argument("--training-only", action="store_true",
                        help="No holdouts or accuracy claim; upstream train images only")
    args = parser.parse_args()
    build(
        "data/raw/matterport-balloon/balloon_dataset.zip",
        args.review,
        "data/raw/matterport-balloon/review/index.json",
        args.output,
        read_json(args.config),
        training_only=args.training_only,
    )


if __name__ == "__main__":
    main()
