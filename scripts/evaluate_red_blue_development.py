"""Thresholded full-scene development check; tiny public set, NOT flight qualification."""
import argparse
import io
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from dtr.data import read_json, sha256
from dtr.runtime import Predictor
from dtr.tracking import iou
from dtr.vision import RED_BLUE_PROFILE, suppress_duplicates, validate_model_profile
from scripts.research_balloon_search import experimental

LABELS = ("red_balloon", "blue_balloon")


def detection_counts(truth, detections):
    """Confidence-ordered one-to-one matching; duplicates and wrong colors count."""
    matched = set()
    counts = {label: dict(tp=0, fp=0, fn=0) for label in LABELS}
    for detection in sorted(detections, key=lambda row: -row["score"]):
        if not detection["accepted"] or detection["label"] not in counts:
            continue
        candidates = [(iou(t["box"], detection["box"]), i) for i, t in enumerate(truth)
                      if i not in matched and t["label"] == detection["label"]]
        overlap, index = max(candidates, default=(0., -1))
        if overlap >= .5:
            matched.add(index)
            counts[detection["label"]]["tp"] += 1
        else:
            counts[detection["label"]]["fp"] += 1
    for i, target in enumerate(truth):
        if i not in matched:
            counts[target["label"]]["fn"] += 1
    return counts


def metrics(counts):
    return {label: dict(**c, precision=c["tp"]/(c["tp"]+c["fp"]) if c["tp"]+c["fp"] else None,
                        recall=c["tp"]/(c["tp"]+c["fn"]) if c["tp"]+c["fn"] else None)
            for label, c in counts.items()}


def evaluate(manifest, scene_review, models, negatives, negative_review):
    doc, review = read_json(manifest), read_json(scene_review)
    if sha256(manifest) != review["manifest_sha256"]:
        raise ValueError("Scene review does not match manifest")
    scenes = defaultdict(list)
    for row in doc["samples"]:
        if row["split"] == "val":
            scenes[row["source_image"]].append(row)
    if set(scenes) != set(review["sources"]):
        raise ValueError("Scene review coverage incomplete")
    archive_path = Path("data/raw/matterport-balloon/balloon_dataset.zip")
    if sha256(archive_path) != doc["source_archive_sha256"]:
        raise ValueError("Archive checksum mismatch")
    queue = read_json(Path(negatives)/"review.json")
    decision = read_json(negative_review)
    if sha256(Path(negatives)/"review.json") != decision["queue_sha256"]:
        raise ValueError("Negative review changed")
    extra = [queue["samples"][i] for i in decision["admit"]]
    results = {}
    for name, path in models.items():
        predictor = Predictor(path, allow_unvalidated=True)
        validate_model_profile(predictor.metadata, RED_BLUE_PROFILE)
        crop_counts = {label: dict(tp=0, fp=0, fn=0) for label in LABELS}
        for row in doc["samples"]:
            if row["split"] != "val":
                continue
            pred = predictor.predict(Path(manifest).parent/row["path"])
            if pred["accepted"] and pred["label"] in crop_counts:
                crop_counts[pred["label"]]["tp" if pred["label"] == row["label"] else "fp"] += 1
            if row["label"] in crop_counts and not (pred["accepted"] and pred["label"] == row["label"]):
                crop_counts[row["label"]]["fn"] += 1
        fits = []
        for row in extra:
            crop = Path(negatives)/row["path"]
            if sha256(crop) != row["sha256"]:
                raise ValueError("Reviewed crop changed")
            pred = predictor.predict(crop)
            fits.append(dict(id=row["id"], label=pred["label"], accepted=pred["accepted"], score=pred["score"]))
        variants = {}
        with zipfile.ZipFile(archive_path) as archive:
            for variant in ("baseline", "multisat", "mser", "mser_components"):
                total = {label: dict(tp=0, fp=0, fn=0) for label in LABELS}
                frames = []
                for source, rows in sorted(scenes.items()):
                    targets = [r for r in rows if r["label"] in LABELS]
                    if sorted(r["annotation_id"] for r in targets) != sorted(review["sources"][source]["target_ids"]):
                        raise ValueError("Reviewed target identities changed")
                    with Image.open(io.BytesIO(archive.read(source))) as opened:
                        image = opened.convert("RGB")
                        rgb = cv2.resize(np.asarray(image), (320, 240), interpolation=cv2.INTER_AREA)
                    truth = [dict(label=r["label"], box=[v*(320/image.width if i % 2 == 0 else 240/image.height)
                                                        for i, v in enumerate(r["box_xyxy"])]) for r in targets]
                    found = []
                    for candidate in experimental(rgb, variant):
                        a, b, c, d = candidate["crop_box"]
                        found.append(dict(candidate, **predictor.predict(rgb[b:d, a:c])))
                    detections = suppress_duplicates(found)
                    counts = detection_counts(truth, detections)
                    for label in LABELS:
                        for key in ("tp", "fp", "fn"):
                            total[label][key] += counts[label][key]
                    frames.append(dict(source=source, truth=truth, counts=counts, detections=detections))
                variants[variant] = dict(metrics=metrics(total), frames=frames)
        results[name] = dict(model_sha256=sha256(path), threshold=predictor.metadata["threshold"],
                             thresholded_crop_validation=metrics(crop_counts),
                             reviewed_negative_training_fit=fits, full_frame=variants)
    return dict(scope="4 reviewed off-domain development photos, not flight qualification",
                manifest_sha256=sha256(manifest), scene_review_sha256=sha256(scene_review),
                negative_review_sha256=sha256(negative_review), script_sha256=sha256(__file__),
                test_evaluated=False, deployment_approved=False, pi_timing_measured=False,
                insufficient_for_94_percent_gate=True, results=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--scene-review", required=True)
    parser.add_argument("--model", action="append", required=True, help="name=path")
    parser.add_argument("--negatives", required=True)
    parser.add_argument("--negative-review", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    models = dict(item.split("=", 1) for item in args.model)
    report = evaluate(args.manifest, args.scene_review, models, args.negatives, args.negative_review)
    args.output.parent.mkdir(exist_ok=True, parents=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({name: dict(crops=result["thresholded_crop_validation"],
                      negative_fit_accepted=sum(r["accepted"] for r in result["reviewed_negative_training_fit"]),
                      full_frame={v: r["metrics"] for v, r in result["full_frame"].items()})
                      for name, result in report["results"].items()}, indent=2))


if __name__ == "__main__":
    main()
