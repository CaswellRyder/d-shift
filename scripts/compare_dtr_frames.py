"""Compare fixed-proposal validation runs; never interpret this as deployment approval."""

import argparse
from pathlib import Path

from dtr.data import read_json, sha256, write_json


def aggregate(classes):
    tp = sum(c["true_positive"] for c in classes.values())
    fp = sum(c["false_positive"] for c in classes.values())
    targets = sum(c["targets"] for c in classes.values())
    return dict(
        true_positive=tp,
        false_positive=fp,
        targets=targets,
        precision=tp / max(1, tp + fp),
        recall=tp / max(1, targets),
        f1=2 * tp / max(1, tp + fp + targets),
    )


def compare(before, after, allow_vision_change=False, allow_budget_change=False):
    for key in (
        "split",
        "frames",
        "image_size",
        "annotation_sha256",
    ):
        if before[key] != after[key]:
            raise ValueError(f"Comparison scope differs: {key}")
    budgets = [
        {
            task: report.get("proposal_limits", {}).get(task, report["proposal_limit"])
            for task in report["tasks"]
        }
        for report in (before, after)
    ]
    budget_changed = budgets[0] != budgets[1]
    if budget_changed and not allow_budget_change:
        raise ValueError("Comparison scope differs: proposal budget")
    vision_keys = ("vision_sha256", "proposal_profile", "duplicate_policy")
    defaults = {"proposal_profile": "v2", "duplicate_policy": "none"}
    vision_changed = any(
        before.get(k, defaults.get(k)) != after.get(k, defaults.get(k)) for k in vision_keys
    )
    if vision_changed and not allow_vision_change:
        raise ValueError("Comparison scope differs: vision configuration")
    if before["split"] != "val" or before["test_evaluated"] or after["test_evaluated"]:
        raise ValueError("This comparison is for development validation only")
    if set(before["tasks"]) != set(after["tasks"]):
        raise ValueError("Task sets differ")
    tasks = {}
    for task, a in before["tasks"].items():
        b = after["tasks"][task]
        if (allow_vision_change or allow_budget_change) and a["model_sha256"] != b["model_sha256"]:
            raise ValueError("Vision ablations require fixed teacher checkpoints")
        if a["threshold"] != b["threshold"]:
            raise ValueError("Acceptance thresholds differ")
        if [f["file"] for f in a["frames"]] != [f["file"] for f in b["frames"]]:
            raise ValueError("Frame lists differ")
        if {k: v["targets"] for k, v in a["classes"].items()} != {
            k: v["targets"] for k, v in b["classes"].items()
        }:
            raise ValueError("Class/target counts differ")
        tasks[task] = dict(
            before=aggregate(a["classes"]),
            after=aggregate(b["classes"]),
            before_model_sha256=a["model_sha256"],
            after_model_sha256=b["model_sha256"],
        )
    return dict(
        scope=(
            "Same validation frames/teachers/threshold; vision ablation, not independent test"
            if allow_vision_change or allow_budget_change
            else "Same validation frames/proposals/threshold; not independent test"
        ),
        vision_changed=vision_changed,
        budget_changed=budget_changed,
        proposal_limits_before=budgets[0],
        proposal_limits_after=budgets[1],
        same_proposal_budget=not budget_changed,
        vision_before={k: before.get(k, defaults.get(k)) for k in vision_keys},
        vision_after={k: after.get(k, defaults.get(k)) for k in vision_keys},
        deployment_approved=False,
        test_evaluated=False,
        tasks=tasks,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--allow-vision-change",
        action="store_true",
        help="Explicit frozen-teacher proposal/postprocessing ablation",
    )
    parser.add_argument(
        "--allow-budget-change",
        action="store_true",
        help="Explicitly compare unequal proposal budgets; fixed teachers required",
    )
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    report = compare(
        read_json(args.before),
        read_json(args.after),
        args.allow_vision_change,
        args.allow_budget_change,
    )
    report.update(
        before_receipt_sha256=sha256(args.before), after_receipt_sha256=sha256(args.after)
    )
    write_json(args.output, report)
    print(report)
