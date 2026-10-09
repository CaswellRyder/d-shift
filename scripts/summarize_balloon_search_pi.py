"""Summarize complete Pi replay trials without counting repeats as accuracy data."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from dtr.data import read_json, sha256, write_json
from scripts.evaluate_red_blue_development import LABELS, detection_counts, metrics
from scripts.pi_red_blue_bench import statistics


def validate_trial(report, rows, inputs):
    n = len(inputs["frames"])
    if (
        report.get("machine") != "armv6l"
        or report.get("camera_used") is not False
        or report.get("deployment_approved") is not False
        or report.get("test_evaluated") is not False
        or report.get("flight_commands") is not None
        or not n
        or report.get("unique_frames") != n
        or report.get("frames") != len(rows)
        or type(report.get("rounds")) is not int
        or not 1 <= report["rounds"] <= 10
        or len(rows) != report["rounds"] * n
    ):
        raise ValueError("Require complete ARMv6 development-only replay")
    first = {}
    for position, row in enumerate(rows):
        repeat, index = divmod(position, n)
        frame = inputs["frames"][index]
        if (
            row["round"] != repeat
            or row["index"] != index
            or row["source"] != frame["source"]
            or row["panel"] != frame["panel"]
        ):
            raise ValueError("Replay ordering or source identity changed")
        if repeat == 0:
            first[index] = row
        elif row["detections"] != first[index]["detections"]:
            raise ValueError("Repeated-frame predictions changed")
        for key in ("processing_ms", "search_ms", "inference_ms", "selection_ms"):
            statistics([row[key]])
    return list(first.values())


def summarize(base, results):
    base, results = Path(base), Path(results)
    inputs = read_json(base / "inputs.json")
    groups = defaultdict(list)
    runs = {}
    for path in sorted(results.glob("*.json")):
        report = read_json(path)
        raw = path.with_suffix(".frames.jsonl")
        if (
            report["frames_log_sha256"] != sha256(raw)
            or report["bundle_sha256"] != sha256(base / "bundle.json")
            or report["inputs_sha256"] != sha256(base / "inputs.json")
            or report["golden_sha256"] != sha256(base / "search-golden.json")
            or report["model_sha256"] != sha256(base / inputs["models"][report["model"]]["path"])
        ):
            raise ValueError("Trial artifact identity mismatch")
        rows = [json.loads(line) for line in raw.read_text().splitlines()]
        unique = validate_trial(report, rows, inputs)
        counts = {}
        for panel in {r["panel"] for r in unique}:
            total = {label: dict(tp=0, fp=0, fn=0) for label in LABELS}
            for row in (r for r in unique if r["panel"] == panel):
                found = detection_counts(inputs["frames"][row["index"]]["truth"], row["detections"])
                for label in LABELS:
                    for key in ("tp", "fp", "fn"):
                        total[label][key] += found[label][key]
            counts[panel] = metrics(total)
        runs[path.stem] = dict(
            report=report, report_sha256=sha256(path), unique_scene_metrics=counts
        )
        groups[(report["model"], report["search"])].append((path.stem, rows))
    if not runs:
        raise ValueError("No completed trials")
    pooled = {}
    for (model, search), trials in groups.items():
        names = [name for name, _ in trials]
        values = [r["processing_ms"] for _, rows in trials for r in rows]
        if any(
            runs[name]["unique_scene_metrics"] != runs[names[0]]["unique_scene_metrics"]
            for name in names
        ):
            raise ValueError("Repeated trials changed scene metrics")
        pooled[model + ":" + search] = dict(
            trials=names,
            timing_samples=len(values),
            timing=statistics(values),
            processing_fps=1000 / statistics(values)["mean_ms"],
            unique_scene_metrics=runs[names[0]]["unique_scene_metrics"],
        )
    return dict(
        runs=runs,
        pooled=pooled,
        script_sha256=sha256(__file__),
        matcher_sha256=sha256("scripts/evaluate_red_blue_development.py"),
        scope="Processing-only replay; unique scenes counted once per model/search, not across repeats",
        test_evaluated=False,
        deployment_approved=False,
        camera_used=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "results", "output"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    report = summarize(args.base, args.results)
    write_json(args.output, report)
    print(report["pooled"])
