"""Freeze a quality cutoff from grouped TRAIN-only predictions, never IMG scenes."""
import argparse
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from scripts.train_balloon_quality import fold_ids, training_rows


def gate_from_oof(rows, report):
    examples = report["examples"]
    if len(examples) != len(rows) or not rows:
        raise ValueError("Quality calibration row coverage mismatch")
    if report.get("test_evaluated") is not False or report.get("deployment_approved") is not False:
        raise ValueError("Require training-only quality predictions")
    for row, example, fold in zip(rows, examples, fold_ids(rows), strict=True):
        if (example["source_group"] != row["source_group"] or example["target"] != row["quality"]
                or example["fold"] != fold or not np.isfinite(example["oof"])
                or not 0 <= example["oof"] <= 1):
            raise ValueError("Quality calibration source, fold, target or prediction changed")
    positive = [e for e in examples if e["target"] >= .5]
    groups = {e["source_group"] for e in positive}
    if len(positive) < 2 or len(groups) < 2:
        raise ValueError("Require multiple localized training targets and source groups")
    # Fixed empirical retention objective, not a confidence bound or flight gate.
    required = int(np.ceil(.94*len(positive)))
    threshold = float(sorted(e["oof"] for e in positive)[len(positive)-required])
    retained = sum(e["oof"] >= threshold for e in positive)
    negative = [e for e in examples if e["target"] < .5]
    return dict(threshold=threshold, comparison="quality >= threshold", retention_target=.94,
                localized_examples=len(positive), localized_source_groups=len(groups),
                localized_retained=retained, empirical_localized_retention=retained/len(positive),
                low_quality_examples=len(negative), low_quality_rejected=sum(e["oof"] < threshold for e in negative),
                rule="Lowest retained grouped out-of-fold score keeping ceil(0.94*N) targets with IoU >= 0.5",
                qualification_claim=False)


def calibrate(model, config, report):
    metadata = read_json(Path(model).with_suffix(".json"))
    if (metadata.get("kind") != "balloon_quality_research" or metadata["sha256"] != sha256(model)
            or metadata["training_config_sha256"] != sha256(config)
            or metadata.get("test_evaluated") is not False or metadata.get("deployment_approved") is not False):
        raise ValueError("Require matching frozen research quality model")
    rows, reviews = training_rows(read_json(config))
    result = gate_from_oof(rows, read_json(report))
    return dict(result, model_sha256=sha256(model), training_config=str(config), training_config_sha256=sha256(config),
                quality_report=str(report), quality_report_sha256=sha256(report), source_reviews=reviews,
                script_sha256=sha256(__file__), test_evaluated=False, deployment_approved=False,
                pi_timing_measured=False,
                caveat="Only head folds held out; backbone saw related imagery. Empirical training retention is not detection recall or qualification.")


def load_gate(path, model):
    saved = read_json(path)
    current = calibrate(model, saved["training_config"], saved["quality_report"])
    if saved != current:
        raise ValueError("Quality gate calibration or identity changed")
    return current["threshold"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "config", "report", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = calibrate(args.model, args.config, args.report)
    write_json(args.output, result)
    print(result)
