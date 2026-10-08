"""Evaluate hash-bound, fully reviewed development scenes, never qualification."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, write_json
from dtr.vision import RED_BLUE_PROFILE, suppress_duplicates, validate_model_profile
from scripts.evaluate_red_blue_development import LABELS, detection_counts, metrics
from scripts.research_balloon_search import experimental, suppress_confirmed_balloon_parts
from scripts.research_predictor import classify_candidates, load_predictor


def reviewed_frames(root, decision_path):
    root = Path(root).resolve()
    queue_path = root / "review.json"
    queue, review = read_json(queue_path), read_json(decision_path)
    if (sha256(queue_path) != review["queue_sha256"]
            or review.get("evaluation_approved") is not True
            or review.get("training_approved") is not False
            or review.get("deployment_approved") is not False):
        raise ValueError("Require matching evaluation-only review")
    admit, exclude = review["admit"], review["exclude"]
    all_ids = admit + exclude
    if (not admit or any(type(i) is not int for i in all_ids)
            or len(set(all_ids)) != len(all_ids)
            or set(all_ids) != set(range(len(queue["frames"])))):
        raise ValueError("Review must decide every frame exactly once")
    frames = []
    for i in admit:
        row = queue["frames"][i]
        path = (root / row["path"]).resolve()
        if row["id"] != i or not path.is_relative_to(root) or sha256(path) != row["sha256"]:
            raise ValueError("Reviewed frame changed or escaped queue")
        with Image.open(path) as im:
            rgb = np.asarray(im.convert("RGB"))
        if rgb.shape != (240, 320, 3):
            raise ValueError("Require reviewed 320x240 processing coordinates")
        for target in row["provisional_truth"]:
            box = target["box"]
            if (target["label"] not in LABELS or len(box) != 4
                    or not all(np.isfinite(v) for v in box)
                    or not 0 <= box[0] < box[2] <= 320
                    or not 0 <= box[1] < box[3] <= 240):
                raise ValueError("Invalid reviewed target")
        frames.append((row, rgb))
    return frames, review


def evaluate(root, review_path, models, search_variants=("baseline", "mser_confirmed_parts")):
    if not search_variants or any(v not in ("baseline", "mser_confirmed_parts", "mser_chromatic", "mser_blue_red") for v in search_variants):
        raise ValueError("Unknown or empty search variants")
    frames, review = reviewed_frames(root, review_path)
    if review["threshold"] != .8 or review["matching_iou"] != .5:
        raise ValueError("Frozen operating point changed")
    results = {}
    for name, path in models.items():
        predictor = load_predictor(path)
        validate_model_profile(predictor.metadata, RED_BLUE_PROFILE)
        if predictor.metadata["threshold"] != review["threshold"]:
            raise ValueError("Model threshold does not match frozen review")
        variants = {}
        for variant in search_variants:
            total = {label: dict(tp=0, fp=0, fn=0) for label in LABELS}
            outputs = []
            for row, rgb in frames:
                found = classify_candidates(predictor, rgb, experimental(rgb, variant))
                detections = suppress_duplicates(found)
                if variant in ("mser_confirmed_parts", "mser_chromatic", "mser_blue_red"):
                    detections = suppress_confirmed_balloon_parts(detections)
                counts = detection_counts(row["provisional_truth"], detections)
                for label in LABELS:
                    for key in ("tp", "fp", "fn"):
                        total[label][key] += counts[label][key]
                outputs.append(dict(id=row["id"], source=row["source"], frame_sha256=row["sha256"],
                                    truth=row["provisional_truth"], counts=counts, detections=detections))
            variants[variant] = dict(metrics=metrics(total), frames=outputs)
        results[name] = dict(model_sha256=sha256(path),
                             runtime="keras_teacher" if Path(path).suffix == ".keras" else "tflite_student",
                             metadata_sha256=sha256(Path(path).with_suffix(".json")),
                             threshold=predictor.metadata["threshold"], full_frame=variants)
    return dict(scope=review["scope"], queue_sha256=review["queue_sha256"],
                review_sha256=sha256(review_path), script_sha256=sha256(__file__),
                search_script_sha256=sha256("scripts/research_balloon_search.py"),
                predictor_script_sha256=sha256("scripts/research_predictor.py"),
                matching_script_sha256=sha256("scripts/evaluate_red_blue_development.py"),
                reviewed_frame_count=len(frames), excluded_frame_ids=review["exclude"],
                independent_recording_sessions_verified=False, test_evaluated=False,
                insufficient_for_94_percent_gate=True, deployment_approved=False,
                pi_timing_measured=False, results=results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", required=True)
    parser.add_argument("--review", required=True)
    parser.add_argument("--model", action="append", required=True, help="name=path (.keras teacher or .tflite student)")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--variant", action="append", choices=("baseline", "mser_confirmed_parts", "mser_chromatic", "mser_blue_red"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    models = dict(item.split("=", 1) for item in args.model)
    if len(models) != len(args.model):
        parser.error("Model names must be unique")
    report = evaluate(args.queue, args.review, models, args.variant or ("baseline", "mser_confirmed_parts"))
    write_json(args.output, report)
    for name, result in report["results"].items():
        print(name, {v: r["metrics"] for v, r in result["full_frame"].items()})
