"""Recompute timing and development counts from hash-bound search experiments."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from scripts.compare_pi_search_paths import METHODS, summarize, validate_identity
from scripts.evaluate_red_blue_development import detection_counts, metrics
from scripts.pi_balloon_search_bench import check_predictions
from scripts.pi_balloon_live_bench import frame_timing


def review(base, golden_path, replay, live):
    golden = read_json(golden_path)
    inputs = read_json(Path(base) / "inputs.json")
    if sha256(Path(base) / "inputs.json") != golden["identity"]["inputs_sha256"]:
        raise ValueError("Different input frames")
    frames = inputs["frames"]
    counts = {}
    for method in METHODS:
        total = defaultdict(lambda: {c: dict(tp=0, fp=0, fn=0)
                                      for c in ("red_balloon", "blue_balloon")})
        for frame, prediction in zip(frames, golden["predictions"][method], strict=True):
            for color, values in detection_counts(frame["truth"], prediction).items():
                for key, value in values.items():
                    total[frame["panel"]][color][key] += value
        counts[method] = {panel: metrics(values) for panel, values in total.items()}
    pooled = defaultdict(list)
    live_rows = defaultdict(list)
    live_elapsed = defaultdict(float)
    live_results = []
    receipts = []
    runtime = None
    for folder in [Path(replay), *map(Path, live)]:
        report = read_json(folder / "report.json")
        validate_identity(report["identity"], golden["identity"])
        if report["machine"] != "armv6l" or report["golden_sha256"] != sha256(golden_path):
            raise ValueError("Wrong hardware or golden reference")
        if report["deployment_approved"] or report["test_evaluated"] or report["flight_commands"] is not None:
            raise ValueError("Experiment authority changed")
        pair = (report["runtime_library_sha256"], report["region_library_sha256"])
        if runtime is not None and runtime != pair:
            raise ValueError("Runtime changed between trials")
        runtime = pair
        if report["mode"] == "replay" and [t["method"] for t in report["trials"]] != [
                "mser", "pixel1", "pixel2", "pixel2", "pixel1", "mser"]:
            raise ValueError("Incomplete balanced replay")
        receipts.append(dict(path=str(folder), sha256=sha256(folder / "report.json")))
        for trial in report["trials"]:
            path = folder / f"{trial['trial']}-{trial['method']}.json"
            if sha256(path) != trial["rows_sha256"]:
                raise ValueError("Changed raw rows")
            rows = read_json(path)["rows"]
            if len(rows) != trial["frames"] or [r["index"] for r in rows] != list(range(len(rows))):
                raise ValueError("Incomplete rows")
            if summarize(rows, trial["elapsed_seconds"]) != trial["summary"]:
                raise ValueError("Reported timing does not match rows")
            if report["mode"] == "replay":
                for row, expected in zip(rows, golden["predictions"][trial["method"]], strict=True):
                    check_predictions(row["detections"], expected)
                pooled[trial["method"]].extend(rows)
            else:
                if not report["camera_used"] or report["preview_enabled"]:
                    raise ValueError("Live scope changed")
                previous = None
                for row in rows:
                    timing = frame_timing(
                        dict(SensorTimestamp=row["sensor_timestamp_ns"]),
                        row["capture_started_boottime_ns"], row["capture_completed_boottime_ns"],
                        row["result_boottime_ns"], previous)
                    for key in ("timestamp_valid", "sensor_timestamp_increasing",
                                "sensor_to_acquire_ms", "sensor_to_result_ms"):
                        if row[key] != timing[key]:
                            raise ValueError("Live freshness does not match timestamps")
                    if timing["timestamp_valid"]:
                        previous = row["sensor_timestamp_ns"]
                live_results.append(dict(path=str(folder), **trial))
                live_rows[trial["method"]].extend(rows)
                live_elapsed[trial["method"]] += trial["elapsed_seconds"]
    replay_results = {method: summarize(pooled[method]) for method in METHODS}
    for method in METHODS:
        replay_results[method]["relative_fps"] = (replay_results[method]["processing_fps"] /
                                                  replay_results["mser"]["processing_fps"])
    # Per-frame counts are never multiplied by timing repetitions.
    return dict(identity=golden["identity"], unique_development_frames=len(frames),
                counts=counts, replay=replay_results, live=live_results,
                live_pooled={method: summarize(rows, live_elapsed[method])
                             for method, rows in live_rows.items()}, receipts=receipts,
                camera_modes_compared=False, live_scenes_matched=False,
                test_evaluated=False, deployment_approved=False, flight_commands=None,
                reviewer_sha256=sha256(__file__), numpy=np.__version__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--golden", required=True)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--live", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    write_json(args.output, review(args.base, args.golden, args.replay, args.live))


if __name__ == "__main__":
    main()
