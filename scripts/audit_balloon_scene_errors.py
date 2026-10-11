"""Attribute frozen development errors without training or changing decisions.

Oracle matching uses truth and is diagnostic ONLY, never an inference policy.
It measures the box coverage ceiling, not achievable model accuracy.
"""
import argparse
from collections import Counter

from dtr.data import read_json, sha256, write_json
from dtr.tracking import iou
from scripts.evaluate_red_blue_development import LABELS, detection_counts, metrics
from scripts.evaluate_reviewed_balloon_scenes import reviewed_frames


def maximum_matching(truth, detections, eligible):
    """Maximum-cardinality one-to-one matching, not independent per-target hits."""
    edges = [[j for j, d in enumerate(detections)
              if eligible(t, d) and iou(t["box"], d["box"]) >= .5] for t in truth]
    owners = {}

    def assign(i, seen):
        for j in edges[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in owners or assign(owners[j], seen):
                owners[j] = i
                return True
        return False

    for i in range(len(truth)):
        assign(i, set())
    return {i: j for j, i in owners.items()}


def analyze_frame(frame):
    truth, rows = frame["truth"], frame["detections"]
    matched, false_positives = set(), []
    # Same ordering, tie break and IoU convention as the frozen evaluator.
    for j in sorted(range(len(rows)), key=lambda j: -rows[j]["score"]):
        d = rows[j]
        if not d["accepted"] or d["label"] not in LABELS:
            continue
        overlaps = [(iou(t["box"], d["box"]), i) for i, t in enumerate(truth)
                    if i not in matched and t["label"] == d["label"]]
        overlap, index = max(overlaps, default=(0., -1))
        if overlap >= .5:
            matched.add(index)
            continue
        same = max((iou(t["box"], d["box"]) for t in truth
                    if t["label"] == d["label"]), default=0.)
        other = max((iou(t["box"], d["box"]) for t in truth
                     if t["label"] != d["label"]), default=0.)
        reason = ("duplicate" if same >= .5 else "wrong_color" if other >= .5
                  else "localization_overlap" if max(same, other) >= .1
                  else "little_or_no_annotated_overlap")
        false_positives.append(dict(index=j, label=d["label"], reason=reason,
                                    same_color_iou=same, other_color_iou=other))
    false_negatives = []
    for i, t in enumerate(truth):
        if i in matched:
            continue
        localized = [d for d in rows if iou(t["box"], d["box"]) >= .5]
        same = [d for d in localized if d["label"] == t["label"]]
        raw = [d for d in same if d["raw_accepted"]]
        accepted = [d for d in same if d["accepted"]]
        reason = ("no_localized_proposal" if not localized
                  else "different_predicted_class" if not same
                  else "below_acceptance_threshold" if not raw
                  else "suppressed" if not accepted else "one_to_one_conflict")
        false_negatives.append(dict(index=i, label=t["label"], reason=reason,
                                    best_proposal_iou=max((iou(t["box"], d["box"]) for d in rows), default=0.)))
    stages = {
        "geometry_oracle": lambda t, d: True,
        "raw_classified": lambda t, d: d["raw_accepted"] and d["label"] == t["label"],
        "selected_classified": lambda t, d: d["accepted"] and d["label"] == t["label"],
    }
    ceilings = {stage: len(maximum_matching(truth, rows, predicate))
                for stage, predicate in stages.items()}
    counts = detection_counts(truth, rows)
    if counts != frame["counts"]:
        raise ValueError("Stored counts disagree with frozen predictions")
    for label in LABELS:
        if (sum(r["label"] == label for r in false_positives) != counts[label]["fp"]
                or sum(r["label"] == label for r in false_negatives) != counts[label]["fn"]):
            raise AssertionError("Error attribution does not conserve detection counts")
    return dict(id=frame["id"], source=frame["source"], counts=counts,
                truth_count=len(truth), matched_upper_bounds=ceilings,
                false_positives=false_positives, false_negatives=false_negatives)


def audit(paths, queue, review):
    admitted, decision = reviewed_frames(queue, review)
    references = {r["id"]: r for r, _ in admitted}
    reports = []
    for path in paths:
        source = read_json(path)
        if (source.get("queue_sha256") != decision["queue_sha256"]
                or source.get("review_sha256") != sha256(review)
                or source.get("test_evaluated") is not False
                or source.get("deployment_approved") is not False
                or source.get("independent_recording_sessions_verified") is not False):
            raise ValueError("Require matching evaluation-only report")
        results = {}
        for name, model in source["results"].items():
            variants = {}
            for variant, result in model["full_frame"].items():
                frames = result["frames"]
                if len(frames) != len(references) or {f["id"] for f in frames} != set(references):
                    raise ValueError("Report frame coverage mismatch")
                for frame in frames:
                    r = references[frame["id"]]
                    if (frame["source"] != r["source"] or frame["frame_sha256"] != r["sha256"]
                            or frame["truth"] != r["provisional_truth"]
                            or frame["vertical_flip"] != r["vertical_flip"]):
                        raise ValueError("Report truth, identity or orientation changed")
                rows = [analyze_frame(f) for f in frames]
                totals = {label: {key: sum(f["counts"][label][key] for f in rows)
                                  for key in ("tp", "fp", "fn")} for label in LABELS}
                if metrics(totals) != result["metrics"]:
                    raise ValueError("Stored aggregate metrics disagree")
                variants[variant] = dict(
                    metrics=metrics(totals),
                    errors={label: {kind: dict(Counter(e["reason"] for f in rows for e in f[kind]
                                                       if e["label"] == label))
                                    for kind in ("false_positives", "false_negatives")} for label in LABELS},
                    truth_count=sum(f["truth_count"] for f in rows),
                    matched_upper_bounds={s: sum(f["matched_upper_bounds"][s] for f in rows)
                                          for s in rows[0]["matched_upper_bounds"]},
                    frames=rows)
            results[name] = dict(model_sha256=model["model_sha256"], variants=variants)
        reports.append(dict(path=str(path), sha256=sha256(path), selection=source["selection"], results=results))
    return dict(scope="Frozen development error attribution; oracle uses truth, not deployable inference",
                script_sha256=sha256(__file__), queue_sha256=decision["queue_sha256"],
                review_sha256=sha256(review), reports=reports, test_evaluated=False,
                deployment_approved=False, independent_recording_sessions_verified=False,
                pi_timing_measured=False, training_performed=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", required=True)
    parser.add_argument("--queue", required=True)
    parser.add_argument("--review", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from pathlib import Path
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    report = audit(args.report, args.queue, args.review)
    write_json(args.output, report)
    for source in report["reports"]:
        for name, model in source["results"].items():
            for variant, result in model["variants"].items():
                print(name, variant, result["matched_upper_bounds"], result["errors"])
