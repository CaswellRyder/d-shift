"""Summarize measured Pi evidence without turning repeated frames into accuracy trials."""
import argparse
import json
from pathlib import Path

from dtr.data import read_json, sha256, write_json
from scripts.evaluate_red_blue_development import detection_counts, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    inputs = read_json(args.inputs)
    truth = {row["source"]: row["truth"] for row in inputs["frames"]}
    records = {}
    for path in sorted(args.results.glob("*.json")):
        row = read_json(path)
        if row["machine"] != "armv6l" or row["input_sha256"] != sha256(args.inputs):
            raise ValueError("Mixed device or inputs")
        raw_path = path.with_suffix(".frames.jsonl")
        frames = [json.loads(line) for line in raw_path.read_text().splitlines()]
        if len(frames) != row["frames"]:
            raise ValueError("Frame log count mismatch")
        entry = dict(report_sha256=sha256(path), frames_sha256=sha256(raw_path), report=row)
        if all("stage_ms" in frame for frame in frames):
            entry["mean_stage_ms"] = {stage: sum(f["stage_ms"][stage] for f in frames)/len(frames)
                                      for stage in frames[0]["stage_ms"]}
        if row["mode"] == "replay":
            first = {}
            for frame in frames:
                first.setdefault(frame["source"], frame)
            if set(first) != set(truth):
                raise ValueError("Incomplete development replay")
            counts = {c: dict(tp=0, fp=0, fn=0) for c in ("red_balloon", "blue_balloon")}
            for source, frame in first.items():
                for label, values in detection_counts(truth[source], frame["observations"]).items():
                    for key, value in values.items():
                        counts[label][key] += value
            entry.update(unique_development_photos=len(first), development_metrics=metrics(counts),
                         processing_fps=1000/row["timing"]["mean_ms"])
        if row["mode"] == "camera":
            entry.update(accepted_observations=sum(o["accepted"] for f in frames for o in f["observations"]),
                         accuracy=None, accuracy_reason="Unlabeled live scene; no precision/recall claim")
        records[path.name] = entry
    if not records:
        raise ValueError("No measured reports")
    write_json(args.output, dict(scope="Device timing plus tiny off-domain development diagnostics",
                                test_evaluated=False, deployment_approved=False,
                                flight_ready=False, results=records))
    for name, entry in records.items():
        row = entry["report"]
        print(name, row["timing"], row.get("observed_loop_fps"), entry.get("development_metrics"))


if __name__ == "__main__":
    main()
