"""Compare hash-checked old/fast Pi replay, requiring identical full detections."""

import argparse
import json
from pathlib import Path

from dtr.data import sha256, write_json
from scripts.summarize_balloon_search_pi import summarize


def compare(summary, results, model="new-views-42"):
    groups = [summary["pooled"][model + ":" + method] for method in ("mser", "mser_fast")]
    if any(len(group["trials"]) < 2 for group in groups):
        raise ValueError("Require at least two trials of each method")
    reference = None
    identity = None
    for group in groups:
        for trial in group["trials"]:
            report = summary["runs"][trial]["report"]
            current = tuple(
                report[key]
                for key in (
                    "model_sha256",
                    "bundle_sha256",
                    "inputs_sha256",
                    "golden_sha256",
                    "runtime_library_sha256",
                    "script_sha256",
                    "opencv",
                    "unique_frames",
                )
            )
            if identity is not None and current != identity:
                raise ValueError("Trial comparison identities changed")
            identity = current
            rows = [
                json.loads(line)
                for line in (Path(results) / (trial + ".frames.jsonl")).read_text().splitlines()
            ]
            first = [row["detections"] for row in rows if row["round"] == 0]
            if reference is not None and first != reference:
                raise ValueError("Full detections changed across search variants/trials")
            reference = first
    old, fast = groups
    return dict(
        model=model,
        exact_full_detection_parity=True,
        unique_frames=len(reference),
        old_processing_fps=old["processing_fps"],
        fast_processing_fps=fast["processing_fps"],
        processing_speedup=fast["processing_fps"] / old["processing_fps"],
        mean_latency_reduction_fraction=1 - fast["timing"]["mean_ms"] / old["timing"]["mean_ms"],
        p95_latency_reduction_fraction=1 - fast["timing"]["p95_ms"] / old["timing"]["p95_ms"],
        trials=old["trials"] + fast["trials"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "results", "output"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    report = summarize(args.base, args.results)
    report["exact_search_comparison"] = compare(report, args.results)
    report["comparison_script_sha256"] = sha256(__file__)
    write_json(args.output, report)
    print(report["exact_search_comparison"])
