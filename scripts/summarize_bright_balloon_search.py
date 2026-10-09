"""Compare changed-policy Pi timing without claiming exact semantic equivalence."""

import argparse
import json
from pathlib import Path

from dtr.data import read_json, sha256, write_json
from scripts.evaluate_red_blue_development import detection_counts
from scripts.pi_red_blue_bench import statistics
from scripts.summarize_balloon_search_pi import summarize


def compare(summary, root, inputs, model="new-views-42"):
    root = Path(root)
    methods = ("mser_direct", "mser_bright")
    groups = [summary["pooled"][model + ":" + method] for method in methods]
    if any(len(g["trials"]) < 2 for g in groups):
        raise ValueError("Require two trials per policy")
    identity, first, diagnostics = None, {}, {}
    for method, group in zip(methods, groups, strict=True):
        all_rows = []
        for trial in group["trials"]:
            report = summary["runs"][trial]["report"]
            if report["search"] != method or report["golden_search"] != (
                "mser" if method == "mser_direct" else "mser_bright"
            ):
                raise ValueError("Policy must use its own explicit host golden")
            if not report.get("region_library_sha256"):
                raise ValueError("Require a bound native library")
            current = tuple(
                report[k]
                for k in (
                    "model_sha256",
                    "bundle_sha256",
                    "inputs_sha256",
                    "golden_sha256",
                    "runtime_library_sha256",
                    "script_sha256",
                    "region_library_sha256",
                    "opencv",
                    "unique_frames",
                )
            )
            if identity is not None and current != identity:
                raise ValueError("Policy comparison identities changed")
            identity = current
            rows = [
                json.loads(s) for s in (root / (trial + ".frames.jsonl")).read_text().splitlines()
            ]
            unique = [r["detections"] for r in rows if r["round"] == 0]
            if method in first and unique != first[method]:
                raise ValueError("Within-policy repeated trials changed predictions")
            first[method] = unique
            all_rows.extend(rows)
        diagnostics[method] = dict(
            search=statistics([r["search_ms"] for r in all_rows]),
            inference=statistics([r["inference_ms"] for r in all_rows]),
            mean_neural_calls=sum(r["neural_calls"] for r in all_rows) / len(all_rows),
        )
    frames = []
    for row, a, b in zip(inputs["frames"], first[methods[0]], first[methods[1]], strict=True):
        ca, cb = detection_counts(row["truth"], a), detection_counts(row["truth"], b)
        frames.append(
            dict(
                source=row["source"],
                panel=row["panel"],
                same_detections=a == b,
                same_counts=ca == cb,
                control_counts=ca,
                bright_counts=cb,
            )
        )
    control, bright = groups
    return dict(
        methods=list(methods),
        processing_speedup=bright["processing_fps"] / control["processing_fps"],
        control_processing_fps=control["processing_fps"],
        bright_processing_fps=bright["processing_fps"],
        mean_latency_reduction_fraction=1
        - bright["timing"]["mean_ms"] / control["timing"]["mean_ms"],
        same_counts_on_each_development_frame=all(r["same_counts"] for r in frames),
        frames_with_changed_detections=sum(not r["same_detections"] for r in frames),
        frames=frames,
        diagnostics=diagnostics,
        general_semantic_equivalence=False,
        known_counterexample="Darker red disk on brighter red field is missed by bright-only search",
        flight_qualified=False,
        deployment_approved=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "results", "output"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    report = summarize(args.base, args.results)
    report["policy_comparison"] = compare(
        report, args.results, read_json(Path(args.base) / "inputs.json")
    )
    report["comparison_script_sha256"] = sha256(__file__)
    write_json(args.output, report)
    print(
        {k: v for k, v in report["policy_comparison"].items() if k not in ("frames", "diagnostics")}
    )
