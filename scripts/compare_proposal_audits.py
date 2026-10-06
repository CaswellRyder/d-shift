"""Train-only geometry regression screen, not detection or deployment approval."""

import argparse
from pathlib import Path

from dtr.data import read_json, sha256, write_json


def compare(before, after):
    for key in ("split", "task", "goal_colors", "image_size", "proposal_limit",
                "iou_threshold", "annotation_sha256", "frames"):
        if key not in before or key not in after or before[key] != after[key]:
            raise ValueError(f"Audit scope differs or missing: {key}")
    if before["split"] != "train" or before["task"] != "goal" or before["goal_colors"] != "all":
        raise ValueError("Require all-color, train-only goal audits")
    if before.get("test_evaluated", True) or after.get("test_evaluated", True):
        raise ValueError("Reserved test evaluation is not accepted")
    if not before["frames"]:
        raise ValueError("No audited frames")
    covered = f"covered{before['proposal_limit']}"
    regressions, deltas = [], {}
    for group in ("counts", "size_counts"):
        a, b = before.get(group, {}), after.get(group, {})
        if not a or set(a) != set(b):
            raise ValueError(f"Audit strata differ or missing: {group}")
        deltas[group] = {}
        for label, baseline in a.items():
            trial = b[label]
            if baseline["targets"] != trial["targets"]:
                raise ValueError(f"Target counts differ: {label}")
            if not 0 <= baseline[covered] <= baseline["targets"] or not (
                0 <= trial[covered] <= trial["targets"]
            ):
                raise ValueError(f"Invalid coverage count: {label}")
            delta = trial[covered] - baseline[covered]
            deltas[group][label] = delta
            if delta < 0:
                regressions.append(dict(group=group, label=label, delta=delta))
    expected = {f"{color} {shape} Goal" for color in ("Orange", "Yellow")
                for shape in ("Circle", "Square", "Triangle")}
    if set(before["counts"]) != expected:
        raise ValueError("All six goal classes are required")
    for report in (before, after):
        for label, counts in report["counts"].items():
            strata = [v for k, v in report["size_counts"].items() if k.startswith(label + "/")]
            if sum(s["targets"] for s in strata) != counts["targets"] or (
                sum(s[covered] for s in strata) != counts[covered]
            ):
                raise ValueError(f"Size strata do not reconcile: {label}")
    orange_gain = sum(v for k, v in deltas["counts"].items() if k.startswith("Orange"))
    return dict(
        scope="Fixed training frames, resolution and candidate budget; geometry only",
        criteria="Orange coverage must improve; no class or class/size stratum may regress",
        eligible_for_validation=orange_gain > 0 and not regressions,
        orange_coverage_gain=orange_gain, regressions=regressions, deltas=deltas,
        deployment_approved=False, test_evaluated=False,
        note="A conservative engineering screen, not statistical significance or model promotion",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    report = compare(read_json(args.before), read_json(args.after))
    report.update(before_sha256=sha256(args.before), after_sha256=sha256(args.after))
    write_json(args.output, report)
    print(report)
    # A rejected experiment is an expected scientific outcome, but blocks shell pipelines.
    return 0 if report["eligible_for_validation"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
