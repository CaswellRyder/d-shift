"""Audit pinned local DTR exports; rebuild grouped splits before crop preparation.

No credentials, network, model predictions, or training. Filename-derived groups are
NOT verified recording sessions. The fixed split policy predates any model scores.
Run once without --prepare, inspect review sheets, then rerun with --prepare.
"""

import argparse
import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from dtr.data import prepare_coco, read_json, sha256, validate, write_json

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/roboflow-dtr-v10/coco"
OUT = ROOT / "data/roboflow-dtr-v10-grouped"
POLICY = {
    "test": "Entire Highbay filename family; no test scores used for model selection",
    "val": "20241002 clips starting 19:20:00 through 19:24:59",
    "guard": "Exclude clips starting 19:19:00-19:19:59 and 19:25:00-19:25:59",
    "train": "Remaining dated clips",
    "near_duplicate": "Remove lower-priority split images at 64-bit dHash distance <= 4",
    "priority": ["test", "val", "train"],
    "caveat": "Clip start times inferred from names; durations and true sessions unknown",
}


def assignment(filename):
    stem = filename.split(".rf.")[0]
    if re.fullmatch(r"Highbay_\d+_jpg", stem):
        return "test", "inferred-Highbay-family"
    match = re.fullmatch(r"20241002_(\d{2})(\d{2})(\d{2})_\d+_jpg", stem)
    if not match:
        return None, "unrecognized-filename"
    hour, minute, second = map(int, match.groups())
    seconds = 3600 * hour + 60 * minute + second
    if 19 * 3600 + 20 * 60 <= seconds < 19 * 3600 + 25 * 60:
        return "val", "inferred-20241002-1920-block"
    if 19 * 3600 + 19 * 60 <= seconds < 19 * 3600 + 26 * 60:
        return None, "temporal-guard"
    return "train", f"inferred-20241002-{hour:02}{minute // 5 * 5:02}-block"


def signature(record):
    return sorted((a["name"], *[round(x, 2) for x in a["bbox"]]) for a in record["boxes"])


def remove_cross_split_near_duplicates(records, distance=4):
    """Conservatively retain test over validation over train; never move held-out images."""
    retained, removed = [], []
    lookup = np.array([int(i).bit_count() for i in range(256)], dtype=np.uint8)
    for split in POLICY["priority"]:
        previous = list(retained)
        hashes = np.array([r["dhash"] for r in previous], dtype=np.uint8).reshape(-1, 8)
        for record in (r for r in records if r["split"] == split):
            if previous:
                differences = lookup[np.bitwise_xor(hashes, record["dhash"])].sum(axis=1)
                closest = int(differences.argmin())
                if int(differences[closest]) <= distance:
                    removed.append(
                        {
                            "source": record["source"],
                            "reason": "cross-split-dhash",
                            "match": previous[closest]["source"],
                            "distance": int(differences[closest]),
                        }
                    )
                    continue
            retained.append(record)
    return retained, removed


def review_sheets(records):
    rng = np.random.default_rng(42)
    by_label = defaultdict(list)
    for record in records:
        if record["split"] == "test":
            continue  # Reserve test even from this representative visual review.
        for box in record["boxes"]:
            by_label[box["name"]].append((record, box))
    for label, examples in sorted(by_label.items()):
        choices = rng.choice(len(examples), size=min(8, len(examples)), replace=False)
        sheet = Image.new("RGB", (4 * 220, 2 * 250), "white")
        draw = ImageDraw.Draw(sheet)
        for i, choice in enumerate(choices):
            record, box = examples[int(choice)]
            with Image.open(RAW / record["source"]) as im:
                x, y, w, h = box["bbox"]
                crop = im.convert("RGB").crop(
                    (
                        max(0, x - 10),
                        max(0, y - 10),
                        min(im.width, x + w + 10),
                        min(im.height, y + h + 10),
                    )
                )
                tile = ImageOps.contain(crop, (215, 215))
            px, py = i % 4 * 220, i // 4 * 250
            sheet.paste(tile, (px, py))
            draw.text((px + 2, py + 217), label, fill="black")
            draw.text((px + 2, py + 231), record["source"].split("/")[-1][:23], fill="black")
        sheet.save(OUT / f"review-{label.replace(' ', '-')}.jpg")


def audit():
    if OUT.exists():
        raise FileExistsError(f"Audit already exists: {OUT}; use --prepare after review")
    records, comparisons = [], {}
    for version in (10, 11):
        root = ROOT / f"data/raw/roboflow-dtr-v{version}/coco"
        stems, count, sizes, labels = set(), 0, Counter(), Counter()
        for folder in ("train", "valid", "test"):
            doc = read_json(root / folder / "_annotations.coco.json")
            categories = {r["id"]: r["name"] for r in doc["categories"]}
            anns = defaultdict(list)
            for ann in doc["annotations"]:
                name = categories[ann["category_id"]]
                labels[name] += 1
                anns[ann["image_id"]].append({"name": name, "bbox": ann["bbox"]})
            for im in doc["images"]:
                filename = im["file_name"]
                source = f"{folder}/{filename}"
                if not (root / source).resolve().is_relative_to(root.resolve()):
                    raise ValueError("Source path escapes root")
                stems.add(filename.split(".rf.")[0])
                count += 1
                sizes[f"{im['width']}x{im['height']}"] += 1
                if version != 10:
                    continue
                split, group = assignment(filename)
                with Image.open(root / source) as opened:
                    image = opened.convert("RGB")
                    digest = hashlib.sha256(image.tobytes()).hexdigest()
                    gray = np.asarray(image.convert("L").resize((9, 8)))
                    dhash = np.packbits(gray[:, 1:] > gray[:, :-1]).tolist()
                records.append(
                    dict(
                        source=source,
                        stem=filename.split(".rf.")[0],
                        split=split,
                        group=group,
                        pixel_sha256=digest,
                        dhash=dhash,
                        boxes=anns[im["id"]],
                        image=im,
                    )
                )
        comparisons[str(version)] = dict(
            images=count, unique_stems=len(stems), dimensions=dict(sizes), annotations=dict(labels)
        )
        if version == 10:
            v10_stems = stems
        else:
            comparisons["shared_stems"] = len(v10_stems & stems)
    dropped, candidates = [], []
    # Duplicate filenames/pixels with conflicting annotations are excluded, not guessed.
    conflicting = set()
    for key in ("stem", "pixel_sha256"):
        grouped = defaultdict(list)
        for record in records:
            grouped[record[key]].append(record)
        for group in grouped.values():
            if any(signature(r) != signature(group[0]) for r in group[1:]):
                conflicting.update(r["source"] for r in group)
    seen_stems, seen_pixels = set(), set()
    priority = {"test": 0, "val": 1, "train": 2, None: 3}
    for record in sorted(records, key=lambda r: (priority[r["split"]], r["source"])):
        reason = None
        if record["source"] in conflicting:
            reason = "conflicting-duplicate-labels"
        elif record["split"] is None:
            reason = record["group"]
        elif record["stem"] in seen_stems or record["pixel_sha256"] in seen_pixels:
            reason = "duplicate-stem-or-pixels"
        if reason:
            dropped.append({"source": record["source"], "reason": reason})
        else:
            candidates.append(record)
            seen_stems.add(record["stem"])
            seen_pixels.add(record["pixel_sha256"])
    retained, near = remove_cross_split_near_duplicates(candidates)
    dropped.extend(near)
    counts = defaultdict(Counter)
    for record in retained:
        counts[record["split"]].update(b["name"] for b in record["boxes"])
    OUT.mkdir(parents=True)
    write_json(
        OUT / "audit.json",
        {
            "source": "https://universe.roboflow.com/cheese-geozd/cats-and-dogs-bmity/dataset/10",
            "attribution": "Cheese / Cats-and-Dogs, Roboflow Universe, version 10",
            "license": "CC BY 4.0",
            "policy": POLICY,
            "versions": comparisons,
            "input_images": len(records),
            "retained_images": len(retained),
            "split_images": dict(Counter(r["split"] for r in retained)),
            "annotations": {k: dict(v) for k, v in counts.items()},
            "excluded": dict(Counter(r["reason"] for r in dropped)),
            "source_archive_sha256": read_json(RAW.parent / "download-receipt.json")[
                "archive_sha256"
            ],
            "qualification": qualification(),
        },
    )
    write_json(OUT / "retained.json", retained)
    write_json(OUT / "excluded.json", dropped)
    review_sheets(retained)
    print((OUT / "audit.json").read_text(), flush=True)


def qualification():
    return {
        "domain": "public DTR arena images with upstream bounding-box labels",
        "source_license": "CC BY 4.0",
        "labels": "upstream labels; representative AI visual check only, not full relabeling",
        "grouping": "inferred filename families/time blocks, NOT verified recording sessions",
        "split_caveat": "Unknown clip duration; residual scene/event correlation possible",
        "deduplication": "source-name, full-pixel SHA and cross-split dHash distance <= 4",
        "backgrounds": "box-exclusion samples; incomplete upstream labels may contaminate negatives",
        "competition_accuracy_established": False,
        "distillation_approved": False,
        "flight_or_pi_qualified": False,
    }


def review_backgrounds():
    manifest = OUT / "balloon/manifest.json"
    rows = [
        r
        for r in read_json(manifest)["samples"]
        if r["split"] != "test" and r["label"] == "background"
    ]
    rng = np.random.default_rng(42)
    sheet = Image.new("RGB", (8 * 160, 5 * 180), "white")
    draw = ImageDraw.Draw(sheet)
    for i, idx in enumerate(rng.choice(len(rows), min(40, len(rows)), replace=False)):
        row = rows[int(idx)]
        with Image.open(manifest.parent / row["path"]) as opened:
            tile = ImageOps.contain(opened.convert("RGB"), (155, 155))
        x, y = i % 8 * 160, i // 8 * 180
        sheet.paste(tile, (x, y))
        draw.text((x + 2, y + 156), f"{i}: {row['split']}", fill="black")
    sheet.save(OUT / "review-backgrounds.jpg")


def prepare():
    records = read_json(OUT / "retained.json")
    # Curated COCO references originals with hard links; no re-encoding or overwrite.
    curated = OUT / "coco"
    if curated.exists():
        raise FileExistsError(curated)
    names = sorted({b["name"] for r in records for b in r["boxes"]})
    categories = [{"id": i, "name": name} for i, name in enumerate(names)]
    ids = {r["name"]: r["id"] for r in categories}
    sessions = {}
    for folder, split in (("train", "train"), ("valid", "val"), ("test", "test")):
        directory = curated / folder
        directory.mkdir(parents=True)
        images, annotations = [], []
        for record in (r for r in records if r["split"] == split):
            filename = record["image"]["file_name"]
            dest = directory / filename
            dest.hardlink_to(RAW / record["source"])
            images.append({**record["image"], "id": len(images)})
            for box in record["boxes"]:
                annotations.append(
                    {
                        "id": len(annotations),
                        "image_id": images[-1]["id"],
                        "category_id": ids[box["name"]],
                        "bbox": box["bbox"],
                    }
                )
            sessions[f"{folder}/{filename}"] = {"split": split, "session": record["group"]}
        write_json(
            directory / "_annotations.coco.json",
            {
                "info": {"description": "Deduplicated, grouped DTR V10; see audit.json"},
                "licenses": [
                    {
                        "id": 1,
                        "name": "CC BY 4.0",
                        "url": "https://creativecommons.org/licenses/by/4.0/",
                    }
                ],
                "categories": categories,
                "images": images,
                "annotations": annotations,
            },
        )
    write_json(OUT / "groups.json", sessions)
    for task in ("balloon", "goal"):
        config = read_json(ROOT / f"configs/{task}.json")
        manifest = Path(prepare_coco(curated, OUT / task, config, OUT / "groups.json"))
        doc = read_json(manifest)
        doc.update(
            qualification=qualification(),
            split_provenance="inferred filename groups with temporal guard and dHash exclusion",
            source_audit_sha256=sha256(OUT / "audit.json"),
        )
        write_json(manifest, doc)
        validate(manifest, config)
        counts = Counter((r["split"], r["label"]) for r in doc["samples"])
        write_json(
            manifest.parent / "counts.json",
            {
                split: {label: counts[split, label] for label in config["classes"]}
                for split in ("train", "val", "test")
            },
        )
        print(task, len(doc["samples"]), dict(counts), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help="After visual review, create crops")
    parser.add_argument("--review-backgrounds", action="store_true")
    args = parser.parse_args()
    if args.review_backgrounds:
        review_backgrounds()
    else:
        prepare() if args.prepare else audit()
