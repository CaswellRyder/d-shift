"""Replay development detections with confidence-ordered nested-box suppression.

Research only. No images, training, threshold changes, new predictions or devices.
Only the existing confirmed-part suppression is reversible; ordinary NMS stays.
"""
import argparse
import math
from pathlib import Path

from dtr.data import read_json, sha256, write_json
from scripts.evaluate_red_blue_development import LABELS, detection_counts, metrics
from scripts.research_balloon_search import suppress_confirmed_balloon_parts


def area(box):
    x, y, z, w = box
    return (z-x)*(w-y)


def nested(a, b):
    smaller, larger = sorted((a, b), key=area)
    x, y, z, w = smaller
    p, q, r, s = larger
    overlap = max(0, min(z, r)-max(x, p))*max(0, min(w, s)-max(y, q))
    return area(smaller) <= .35*area(larger) and overlap >= .95*area(smaller)


def before_parts(observations):
    rows = [dict(row) for row in observations]
    for row in rows:
        box = row["box"]
        if (len(box) != 4 or not all(math.isfinite(v) for v in box)
                or not 0 <= box[0] < box[2] <= 320 or not 0 <= box[1] < box[3] <= 240
                or not math.isfinite(row["score"]) or not 0 <= row["score"] <= 1):
            raise ValueError("Invalid saved scan box or confidence")
        if row.get("suppression_reason") == "confirmed_balloon_part":
            if row["accepted"] or not row.get("suppressed") or row["label"] not in LABELS or row["score"] < .8:
                raise ValueError("Invalid saved confirmed-part suppression")
            row.update(accepted=True, suppressed=False)
            row.pop("suppression_reason")
    return rows


def confidence_ordered(observations):
    """The stronger same-color box wins; ties keep the larger parent as before."""
    rows = before_parts(observations)
    accepted = [i for i, r in enumerate(rows) if r["accepted"] and r["label"] in LABELS]
    accepted.sort(key=lambda i: (-rows[i]["score"], -area(rows[i]["box"]), i))
    kept = []
    for index in accepted:
        row = rows[index]
        winner = next((j for j in kept if rows[j]["label"] == row["label"]
                       and nested(row["box"], rows[j]["box"])), None)
        if winner is None:
            kept.append(index)
        else:
            row.update(accepted=False, suppressed=True,
                       suppression_reason="confidence_ordered_containment", suppressed_by_index=winner)
    return rows


def compare(report):
    if report.get("test_evaluated") is not False or report.get("deployment_approved") is not False:
        raise ValueError("Require explicit development-only evidence")
    results = {}
    for name, model in report["results"].items():
        if model.get("threshold") != .8:
            raise ValueError("Require frozen 0.8 threshold")
        evidence = model["full_frame"]["mser_confirmed_parts"]
        variants = {}
        for kind in ("parent_first", "confidence_ordered"):
            total = {label: dict(tp=0, fp=0, fn=0) for label in LABELS}
            frames = []
            for frame in evidence["frames"]:
                restored = before_parts(frame["detections"])
                replayed = suppress_confirmed_balloon_parts(restored)
                if [r["accepted"] for r in replayed] != [r["accepted"] for r in frame["detections"]]:
                    raise ValueError("Saved parent suppression cannot be reproduced exactly")
                rows = replayed if kind == "parent_first" else confidence_ordered(frame["detections"])
                counts = detection_counts(frame["truth"], rows)
                for label in LABELS:
                    for key in ("tp", "fp", "fn"):
                        total[label][key] += counts[label][key]
                frames.append(dict(frame, detections=rows, counts=counts))
            variants[kind] = dict(metrics=metrics(total), frames=frames)
        if variants["parent_first"]["metrics"] != evidence["metrics"]:
            raise ValueError("Original metrics do not reproduce")
        results[name] = dict(model_sha256=model["model_sha256"], threshold=.8, full_frame=variants)
    return dict(scope="Post-hoc development replay; not independent accuracy or live latency",
                results=results, test_evaluated=False, deployment_approved=False,
                pi_timing_measured=False, new_neural_evaluations=0,
                script_sha256=sha256(__file__),
                original_suppression_script_sha256=sha256("scripts/research_balloon_search.py"),
                matching_script_sha256=sha256("scripts/evaluate_red_blue_development.py"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = compare(read_json(args.report))
    result["source_report_sha256"] = sha256(args.report)
    write_json(args.output, result)
    for name, model in result["results"].items():
        print(name, {k: v["metrics"] for k, v in model["full_frame"].items()})
