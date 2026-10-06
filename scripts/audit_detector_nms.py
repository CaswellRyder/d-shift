"""Audit extra duplicate suppression on cached detections; never changes the runtime."""

import argparse
from pathlib import Path

from dtr.data import read_json, sha256, write_json
from dtr.detector_metrics import summarize
from dtr.tracking import iou


def suppress(predictions, threshold=.7):
    kept, removed = [], []
    for prediction in sorted(predictions, key=lambda p: -p["score"]):
        if any(iou(prediction["box"], other["box"]) > threshold for other in kept):
            removed.append(prediction)
        else:
            kept.append(prediction)
    return kept, removed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--policy", choices=["cross_class", "nested"], default="cross_class")
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    report = read_json(args.report)
    if report["split"] != "val" or report["metrics"]["test_evaluated"]:
        raise ValueError("Development-validation reports only")
    baseline = summarize(report["frames"], report["standard"])
    frames, removed = [], []
    for frame in report["frames"]:
        if args.policy == "nested":
            from dtr.vision import suppress_duplicates
            rows = suppress_duplicates([dict(p, accepted=True) for p in frame["predictions"]])
            kept = [p for p in rows if p["accepted"]]
            suppressed = [p for p in rows if not p["accepted"]]
        else:
            kept, suppressed = suppress(frame["predictions"])
        frames.append(dict(frame, predictions=kept))
        removed.extend(dict(file=frame["file"], prediction=p) for p in suppressed)
    candidate = summarize(frames, report["standard"])
    deltas = {name: {stat: candidate["classes"][name][stat] - values[stat]
                     for stat in ("true_positive", "false_positive", "precision", "recall")}
              for name, values in baseline["classes"].items()}
    result = dict(report_sha256=sha256(args.report), policy=args.policy,
                  nms_iou=.7 if args.policy == "cross_class" else None, baseline=baseline,
                  candidate=candidate, per_class_delta=deltas, suppressed=removed,
                  selected=False, deployment_approved=False, test_evaluated=False,
                  scope="Cached development-only NMS ablation; no runtime change or promotion")
    write_json(args.output, result)
    print(dict(suppressed=len(removed), per_class_delta=deltas,
               development_passed=candidate["development_passed"], selected=False))


if __name__ == "__main__":
    main()
