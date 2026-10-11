"""Attribute development misses from saved detections; never tune or train on them."""
import argparse
from collections import Counter
from pathlib import Path

from dtr.data import read_json, sha256, write_json
from dtr.tracking import iou


def attribute(frame):
    truth, detections = frame["truth"], frame["detections"]
    matched = set()
    for detection in sorted(detections, key=lambda d: -d["score"]):
        if not detection["accepted"]:
            continue
        candidates = [(iou(t["box"], detection["box"]), i) for i, t in enumerate(truth)
                      if i not in matched and t["label"] == detection["label"]]
        overlap, index = max(candidates, default=(0., -1))
        if overlap >= .5:
            matched.add(index)
    rows = []
    for i, target in enumerate(truth):
        eligible = [d for d in detections if iou(target["box"], d["box"]) >= .5]
        correct = [d for d in eligible if d["label"] == target["label"]]
        if i in matched:
            reason = "detected"
        elif not eligible:
            reason = "no_localized_proposal"
        elif not correct:
            reason = "wrong_class_on_localized_proposals"
        elif any(d["accepted"] for d in correct):
            reason = "one_to_one_assignment_conflict"
        elif any(d.get("suppressed", False) for d in correct):
            reason = "suppression_involved"
        else:
            reason = "correct_class_below_threshold"
        rows.append(dict(label=target["label"], box=target["box"], reason=reason,
                         best_proposal_iou=max((iou(target["box"], d["box"]) for d in detections), default=0.),
                         best_correct_class_score=max((d["score"] for d in correct), default=None)))
    return rows


def audit(report):
    results = {}
    for name, model in report["results"].items():
        variants = {}
        for variant, evidence in model["full_frame"].items():
            frames = [dict(source=f["source"], targets=attribute(f)) for f in evidence["frames"]]
            counts = Counter((t["label"], t["reason"]) for f in frames for t in f["targets"])
            variants[variant] = dict(counts={f"{label}/{reason}": n for (label, reason), n in sorted(counts.items())},
                                     frames=frames, detection_metrics=evidence["metrics"])
        results[name] = dict(model_sha256=model["model_sha256"], variants=variants)
    return dict(scope="Development error attribution only; no model or threshold changes",
                deployment_approved=False, qualification_established=False, results=results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit(read_json(args.report))
    result.update(source_report_sha256=sha256(args.report), script_sha256=sha256(__file__))
    write_json(args.output, result)
    for name, model in result["results"].items():
        print(name, {v: r["counts"] for v, r in model["variants"].items()})
