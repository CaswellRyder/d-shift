"""Rank whole raw-detector checkpoints by the weakest required class metric; never promote."""

import argparse
import hashlib
import json
from pathlib import Path

from dtr.data import read_json, sha256, write_json
from dtr.detector_metrics import summarize


def compare(reports):
    if not reports:
        raise ValueError("No reports supplied")
    reference, rows, seen = None, [], set()
    for path, report in reports:
        if (report["split"] != "val" or report.get("experiment")
                or report["metrics"]["test_evaluated"] or not report["frames"]):
            raise ValueError("Only complete raw-detector development reports may be compared")
        truth = sorted([dict(file=f["file"], image_size=f["image_size"], truth=f["truth"])
                        for f in report["frames"]], key=lambda f: f["file"])
        if len({f["file"] for f in truth}) != len(truth):
            raise ValueError("Duplicate evaluation frames")
        identity = dict(standard=report["standard"], input_width=report["detector_input_width"],
                        device=report.get("device"),
                        annotation_sha256=report["annotation_sha256"],
                        truth_sha256=hashlib.sha256(json.dumps(truth, sort_keys=True).encode()).hexdigest())
        if reference is not None and identity != reference:
            raise ValueError("Cannot compare different standards, inputs, labels or frame sets")
        reference = identity
        if report["model_sha256"] in seen:
            raise ValueError("Repeated checkpoint")
        seen.add(report["model_sha256"])
        metrics = summarize(report["frames"], report["standard"])
        ratios = [value / report["standard"][f"minimum_{metric}_per_class"]
                  for row in metrics["classes"].values()
                  for metric, value in row.items() if metric in ("precision", "recall")]
        sufficient = all(row["targets"] >= report["standard"]["minimum_targets_per_class"]
                         for row in metrics["classes"].values())
        rows.append(dict(report=str(path), model_sha256=report["model_sha256"],
                         epoch=report["epoch"], sufficient_targets=sufficient,
                         weakest_requirement_ratio=min(ratios) if sufficient else 0,
                         macro_f1=sum(c["f1"] for c in metrics["classes"].values()) /
                         len(metrics["classes"]), metrics=metrics))
    rows.sort(key=lambda r: (-r["weakest_requirement_ratio"], -r["macro_f1"], str(r["model_sha256"])))
    return dict(comparison=reference, ranking=rows,
                ranking_rule="Minimum per-class precision/required-precision or recall/required-recall; macro F1 breaks ties",
                all_requirements_met_by_top_checkpoint=rows[0]["metrics"]["development_passed"],
                promotion_allowed=False, deployment_approved=False, test_evaluated=False,
                scope="Development-only ranking of whole checkpoints, not independent qualification")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    reports = [(path, read_json(path)) for path in args.reports]
    standard_path = Path("configs/goal-detection-standard.json")
    standard, digest = read_json(standard_path), sha256(standard_path)
    if any(report["standard"] != standard or report["standard_sha256"] != digest
           for _, report in reports):
        raise ValueError("Reports differ from the current recorded acceptance standard")
    result = compare(reports)
    result["source_report_sha256"] = {path: sha256(path) for path in args.reports}
    result["script_sha256"] = sha256(__file__)
    write_json(args.output, result)
    print([dict(epoch=r["epoch"], weakest_requirement_ratio=r["weakest_requirement_ratio"],
                development_passed=r["metrics"]["development_passed"]) for r in result["ranking"]])


if __name__ == "__main__":
    main()
