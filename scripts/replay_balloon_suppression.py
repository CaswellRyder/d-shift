"""Replay post-classification suppression on stored development-scene reports.

Rebuilds each frame's raw classified candidates from a report's saved
detections, checks that the current rule reproduces the saved counts, then
applies merged-parent splitting. No model, image or test pixels are read;
development reports only, not qualification.
"""
import argparse
import json
from pathlib import Path

from dtr.data import sha256
from dtr.vision import suppress_duplicates
from scripts.evaluate_red_blue_development import LABELS, detection_counts
from scripts.research_balloon_search import split_merged_balloon_parents, suppress_confirmed_balloon_parts

DERIVED = ("raw_accepted", "suppressed", "suppressed_by", "rejection_reason", "suppression_reason")


def raw(detections):
    return [dict({k: v for k, v in d.items() if k not in DERIVED}, accepted=d["raw_accepted"])
            for d in detections]


def replay(frames, split):
    total = {label: dict(tp=0, fp=0, fn=0) for label in LABELS}
    changed = []
    for frame in frames:
        rows = suppress_duplicates(raw(frame["detections"]))
        if split:
            rows = split_merged_balloon_parents(rows)
        rows = suppress_confirmed_balloon_parts(rows)
        counts = detection_counts(frame["truth"], rows)
        for label in LABELS:
            for key in ("tp", "fp", "fn"):
                total[label][key] += counts[label][key]
        if counts != frame["counts"]:
            changed.append(dict(frame=frame.get("id", frame.get("source")), before=frame["counts"], after=counts))
    return total, changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+")
    parser.add_argument("--variant", default="mser_confirmed_parts")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = []
    for path in args.reports:
        report = json.loads(Path(path).read_text())
        for model, result in report["results"].items():
            variant = result.get("full_frame", {}).get(args.variant)
            if not variant or not all("raw_accepted" in d for f in variant["frames"] for d in f["detections"]):
                continue
            current, drift = replay(variant["frames"], split=False)
            split, changed = replay(variant["frames"], split=True)
            rows.append(dict(report=str(path), report_sha256=sha256(path), model=model,
                             model_sha256=result["model_sha256"], reproduced=not drift,
                             current=current, split=split, changed_frames=changed))
    doc = dict(purpose="Offline replay of merged-parent splitting on stored development reports; "
                       "NOT qualification and no runtime change",
               variant=args.variant, script_sha256=sha256(__file__),
               rule_script_sha256=sha256("scripts/research_balloon_search.py"),
               test_evaluated=False, deployment_approved=False, rows=rows)
    Path(args.output).write_text(json.dumps(doc, indent=2) + "\n")
    for row in rows:
        delta = {label[0]: "/".join(f"{row['split'][label][k]-row['current'][label][k]:+d}" for k in ("tp", "fp", "fn"))
                 for label in LABELS}
        print(Path(row["report"]).name[:48].ljust(48), row["model"][:16].ljust(16),
              "ok" if row["reproduced"] else "DRIFT", delta, len(row["changed_frames"]))


if __name__ == "__main__":
    main()
