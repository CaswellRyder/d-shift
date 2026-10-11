"""Replay frozen development predictions with a training-calibrated oval bound.

No new proposals, class predictions, thresholds or image-based training. Default
selection is reproduced exactly before comparing the opt-in calibrated policy.
"""
import argparse
import math
from pathlib import Path

from dtr.data import read_json, sha256, write_json
from dtr.vision import suppress_duplicates
from scripts.audit_balloon_pipeline_errors import audit
from scripts.evaluate_red_blue_development import LABELS, detection_counts, metrics
from scripts.pi_balloon_search_bench import select_detections
from scripts.research_balloon_search import split_merged_balloon_parents, suppress_confirmed_balloon_parts


SELECTION_KEYS = ("accepted", "raw_accepted", "suppressed", "suppressed_by",
                  "rejection_reason", "suppression_reason")


def raw_candidates(rows):
    result = []
    for row in rows:
        if type(row.get("raw_accepted")) is not bool:
            raise ValueError("Raw acceptance is missing or invalid")
        candidate = {k:v for k,v in row.items() if k not in SELECTION_KEYS}
        candidate["accepted"] = row["raw_accepted"]
        result.append(candidate)
    return result


def calibrated_selection(rows, max_aspect):
    if not math.isfinite(max_aspect) or not 1.5 <= max_aspect <= 3.:
        raise ValueError("Calibrated aspect bound is outside candidate geometry")
    selected = suppress_duplicates(rows)
    selected = split_merged_balloon_parents(selected, max_aspect=max_aspect)
    return suppress_confirmed_balloon_parts(selected)


def compare(report_path, calibration_path, queue, review):
    # Verifies saved per-frame evidence and counts against hash-bound review.
    audit([report_path], queue, review)
    source, calibration = read_json(report_path), read_json(calibration_path)
    if (source["selection"] != "split_merged"
            or calibration.get("schema") != "balloon-split-geometry-v1"
            or calibration.get("training_only") is not True
            or calibration.get("test_evaluated") is not False
            or calibration.get("deployment_approved") is not False
            or calibration.get("quantile") != .95):
        raise ValueError("Require fixed training-only calibration and split_merged control")
    results = {}
    for name, model in source["results"].items():
        if model["threshold"] != .8:
            raise ValueError("Classifier threshold changed")
        variants = {}
        for variant, saved in model["full_frame"].items():
            if variant not in ("mser_confirmed_parts", "mser_instance_split"):
                raise ValueError("Require frozen baseline or watershed proposals")
            rows = []
            total = {label: dict(tp=0, fp=0, fn=0) for label in LABELS}
            for frame in saved["frames"]:
                raw = raw_candidates(frame["detections"])
                control = select_detections(raw, "split_merged")
                if (len(control) != len(frame["detections"])
                        or any(any(a.get(k) != b.get(k) for k in SELECTION_KEYS)
                               for a,b in zip(control, frame["detections"], strict=True))):
                    raise ValueError("Frozen selection could not be reproduced")
                changed = calibrated_selection(raw, calibration["max_aspect"])
                counts = detection_counts(frame["truth"], changed)
                for label in LABELS:
                    for k in total[label]:
                        total[label][k] += counts[label][k]
                rows.append(dict(frame, detections=changed, counts=counts))
            variants[variant] = dict(control_metrics=saved["metrics"], metrics=metrics(total), frames=rows)
        results[name] = dict(model_sha256=model["model_sha256"], threshold=.8,
                             metadata_sha256=model["metadata_sha256"], runtime=model["runtime"], full_frame=variants)
    return dict(results=results, selection="train_calibrated_split", max_aspect=calibration["max_aspect"],
                source_report_sha256=sha256(report_path), calibration_sha256=sha256(calibration_path),
                queue_sha256=source["queue_sha256"], review_sha256=source["review_sha256"],
                script_sha256=sha256(__file__), selection_script_sha256=sha256("scripts/research_balloon_search.py"),
                duplicate_script_sha256=sha256("src/dtr/vision.py"),
                scope="Post-hoc development replay with training-derived oval bound; no independent qualification",
                test_evaluated=False, deployment_approved=False, new_neural_evaluations=0,
                independent_recording_sessions_verified=False, pi_timing_measured=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("report", "calibration", "queue", "review", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = compare(args.report, args.calibration, args.queue, args.review)
    write_json(args.output, report)
    print({name:{v:r["metrics"] for v,r in result["full_frame"].items()}
           for name,result in report["results"].items()})
