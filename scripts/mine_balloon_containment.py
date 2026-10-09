"""Quarantine unused TRAIN containment cases for review; never admit or fit them."""

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
from dtr.runtime import Predictor
from dtr.tracking import iou
from dtr.vision import suppress_duplicates
from scripts.audit_balloon_pair_supervision import BOOTSTRAP_SHA, preference
from scripts.audit_public_balloon_export import bounded_read, source_group
from scripts.compare_balloon_containment import area, nested
from scripts.prepare_indoor_balloon_review import (
    ARCHIVE_SHA,
    MAP,
    family,
    fingerprint,
    holdout_distance,
)
from scripts.research_balloon_search import experimental
from scripts.train_balloon_quality import training_rows


def eligible_images(docs, used):
    forbidden = {source_group(r["file_name"]) for s in ("valid", "test") for r in docs[s]["images"]}
    groups = {}
    for row in sorted(docs["train"]["images"], key=lambda r: r["file_name"]):
        name = row["file_name"]
        group = source_group(name)
        if family(name) and group not in forbidden | used:
            groups.setdefault(group, row)
    return list(groups.values())


def provisional_targets(image, annotations, categories):
    """These are upstream provisional boxes, not accepted review labels."""
    width, height = image["width"], image["height"]
    if min(width, height) < 1:
        raise ValueError("Invalid source dimensions")
    truth = []
    for annotation in annotations:
        label = MAP.get(categories[annotation["category_id"]])
        x, y, w, h = annotation["bbox"]
        if (
            label is None
            or not np.isfinite([x, y, w, h]).all()
            or min(w, h) <= 0
            or x < 0
            or y < 0
            or x + w > width
            or y + h > height
        ):
            raise ValueError("Unsupported annotation class or invalid geometry")
        truth.append(
            dict(
                id=annotation["id"],
                label=label,
                box=[
                    x * 320 / width,
                    y * 240 / height,
                    (x + w) * 320 / width,
                    (y + h) * 240 / height,
                ],
            )
        )
    return truth


def pair_choices(observations, truth):
    found = suppress_duplicates(observations)
    pairs = []
    for i, a in enumerate(found):
        for j, b in enumerate(found[i + 1 :], i + 1):
            if not (
                a["accepted"]
                and b["accepted"]
                and a["label"] == b["label"]
                and nested(a["box"], b["box"])
            ):
                continue
            (pi, parent), (ci, child) = sorted(
                ((i, a), (j, b)), key=lambda p: area(p[1]["box"]), reverse=True
            )
            target, reason = preference(parent, child, truth)
            if reason == "different_targets":
                matches = []
                for candidate in (parent, child):
                    scores = sorted(
                        [
                            (iou(candidate["box"], t["box"]), t["id"])
                            for t in truth
                            if t["label"] == candidate["label"]
                        ],
                        reverse=True,
                    )
                    matches.append(scores[0])
                if min(v[0] for v in matches) >= 0.5:
                    target = dict(
                        choice="keep_both",
                        parent_iou=matches[0][0],
                        child_iou=matches[1][0],
                        target_ids=[v[1] for v in matches],
                    )
            if target is not None:
                pairs.append(
                    dict(
                        target,
                        label=parent["label"],
                        parent_index=pi,
                        child_index=ci,
                        parent_box=parent["box"],
                        child_box=child["box"],
                        parent_score=parent["score"],
                        child_score=child["score"],
                    )
                )
    return pairs


def mine(archive_path, quality_config, model_paths, output, max_sources=800, max_review=48):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if (
        type(max_sources) is not int
        or not 1 <= max_sources <= 800
        or type(max_review) is not int
        or not 1 <= max_review <= 96
    ):
        raise ValueError("Invalid bounded acquisition limits")
    qconfig = read_json(quality_config)
    rows, receipts = training_rows(qconfig)
    if sha256(qconfig["manifest"]) != BOOTSTRAP_SHA or sha256(archive_path) != ARCHIVE_SHA:
        raise ValueError("Source identity changed")
    base = read_json(qconfig["manifest"])
    held_hashes = {r["source_sha256"] for r in base["samples"] if r["split"] != "train"}
    used = {r["source_group"] for r in rows}
    prior = []
    for source in qconfig["sources"]:
        root = Path(source["queue"]).resolve()
        for frame in read_json(root / "review.json")["frames"]:
            path = (root / frame["path"]).resolve()
            if not path.is_relative_to(root) or sha256(path) != frame["sha256"]:
                raise ValueError("Prior training pixels changed")
            with Image.open(path) as im:
                prior.append(fingerprint(np.asarray(im.convert("RGB"))))
    models = {name: Predictor(path, allow_unvalidated=True) for name, path in model_paths.items()}
    for model in models.values():
        if (
            model.metadata["classes"] != ["background", "red_balloon", "blue_balloon"]
            or model.metadata["size"] != 64
            or model.metadata["threshold"] != 0.8
            or model.metadata["task"] != "balloon"
        ):
            raise ValueError("Unexpected parent classifier contract")
    output.mkdir(parents=True)
    (output / "frames").mkdir()
    rejected = Counter()
    seen = set()
    candidates = []
    scanned = 0
    with zipfile.ZipFile(archive_path) as archive:
        docs = {
            s: json.loads(bounded_read(archive, f"{s}/_annotations.coco.json", 50_000_000))
            for s in ("train", "valid", "test")
        }
        eligible = eligible_images(docs, used)
        categories = {r["id"]: r["name"] for r in docs["train"]["categories"]}
        annotations = defaultdict(list)
        for a in docs["train"]["annotations"]:
            annotations[a["image_id"]].append(a)
        for index, source in enumerate(eligible[:max_sources]):
            try:
                truth = provisional_targets(source, annotations[source["id"]], categories)
            except ValueError:
                rejected["invalid_annotations"] += 1
                continue
            if not truth:
                rejected["no_annotations"] += 1
                continue
            with Image.open(
                io.BytesIO(bounded_read(archive, "train/" + source["file_name"], 20_000_000))
            ) as im:
                im = im.convert("RGB")
                if im.size != (source["width"], source["height"]):
                    raise ValueError("Image dimensions differ from annotation")
                digest = hashlib.sha256(im.tobytes()).hexdigest()
                rgb = cv2.resize(np.asarray(im), (320, 240), interpolation=cv2.INTER_AREA)
            if digest in held_hashes or digest in seen:
                rejected["duplicate_or_reserved_hash"] += 1
                continue
            seen.add(digest)
            if prior and holdout_distance(rgb, prior) <= 8:
                rejected["near_prior_training"] += 1
                continue
            scanned += 1
            proposals = experimental(rgb, "mser")
            merged = {}
            for name, model in models.items():
                observations = []
                for proposal in proposals:
                    a, b, c, d = proposal["crop_box"]
                    observations.append(dict(proposal, **model.predict(rgb[b:d, a:c])))
                for pair in pair_choices(observations, truth):
                    key = (pair["parent_index"], pair["child_index"], pair["choice"], pair["label"])
                    row = merged.setdefault(key, dict(pair, observed_by={}))
                    row["observed_by"][name] = dict(
                        parent_score=pair["parent_score"], child_score=pair["child_score"]
                    )
            if merged:
                relative = f"frames/{index:04d}.png"
                Image.fromarray(rgb).save(output / relative)
                for pair in merged.values():
                    candidates.append(
                        dict(
                            pair,
                            source="train/" + source["file_name"],
                            source_group=source_group(source["file_name"]),
                            source_sha256=digest,
                            frame_path=relative,
                            frame_sha256=sha256(output / relative),
                            provisional_truth=truth,
                            family=family(source["file_name"]),
                        )
                    )
            if scanned % 25 == 0:
                print(
                    dict(scanned=scanned, eligible_considered=index + 1, pairs=len(candidates)),
                    flush=True,
                )

    # Prioritize the missing blue-child cases, then distinct-instance cases.
    def priority(r):
        return (
            0
            if r["choice"] == "child" and r["label"] == "blue_balloon"
            else 1
            if r["choice"] == "keep_both"
            else 2
            if r["label"] == "blue_balloon"
            else 3,
            r["source"],
            r["parent_index"],
            r["child_index"],
        )

    selected = []
    source_choices = set()
    for row in sorted(candidates, key=priority):
        key = (row["source_group"], row["choice"], row["label"])
        if key not in source_choices and len(selected) < max_review:
            selected.append(dict(row, id=len(selected)))
            source_choices.add(key)
    report = dict(
        samples=selected,
        eligible_sources=len(eligible),
        considered_sources=min(max_sources, len(eligible)),
        scanned_sources=scanned,
        exclusions=dict(rejected),
        all_candidate_pairs=len(candidates),
        all_pair_counts=dict(Counter(r["label"] + ":" + r["choice"] for r in candidates)),
        selected_pair_counts=dict(Counter(r["label"] + ":" + r["choice"] for r in selected)),
        archive_sha256=ARCHIVE_SHA,
        base_manifest_sha256=BOOTSTRAP_SHA,
        quality_config_sha256=sha256(quality_config),
        prior_reviews=receipts,
        models={
            k: dict(model_sha256=sha256(p), metadata_sha256=sha256(Path(p).with_suffix(".json")))
            for k, p in model_paths.items()
        },
        script_sha256=sha256(__file__),
        review_required=True,
        training_approved=False,
        test_evaluated=False,
        deployment_approved=False,
        caveat="TRAIN quarantine. Prior-training perceptual dedup and reserved exact hashes only; no held-out pixels read. Not independent sessions.",
    )
    write_json(output / "review.json", report)
    for start in range(0, len(selected), 6):
        sheet = Image.new("RGB", (960, 548), "#dddddd")
        draw = ImageDraw.Draw(sheet)
        for index, row in enumerate(selected[start : start + 6]):
            with Image.open(output / row["frame_path"]) as im:
                marked = im.copy()
            mark = ImageDraw.Draw(marked)
            for t in row["provisional_truth"]:
                mark.rectangle(t["box"], outline="white", width=1)
            mark.rectangle(row["parent_box"], outline="lime", width=2)
            mark.rectangle(row["child_box"], outline="magenta", width=2)
            x, y = index % 3 * 320, index // 3 * 274
            sheet.paste(marked, (x, y + 32))
            draw.text((x, y), f"{row['id']} {row['label']} {row['choice']}", fill="black")
            draw.text(
                (x, y + 15), f"IoU P{row['parent_iou']:.2f} C{row['child_iou']:.2f}", fill="black"
            )
        sheet.save(output / f"pairs-{start:03d}.jpg")
    return {k: v for k, v in report.items() if k not in ("samples", "models", "prior_reviews")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(
        mine(
            "data/raw/engdes2-red-blue-v1-20261008/dataset.zip",
            "configs/balloon-quality-training-20261008.json",
            {
                f"seed{seed}": f"runs/balloon-red-blue-new-views-fixed-teacher{suffix}-20261008/student.fp32.tflite"
                for seed, suffix in ((42, ""), (43, "-seed43"))
            },
            args.output,
        )
    )
