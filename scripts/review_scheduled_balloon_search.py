"""Verify scheduler decisions from timestamps; static replay is not temporal recall."""

import argparse
import json
from pathlib import Path

from dtr.data import read_json, sha256, write_json
from dtr.research_search_schedule import SearchSchedule
from scripts.evaluate_red_blue_development import detection_counts
from scripts.pi_balloon_search_bench import check_predictions
from scripts.pi_red_blue_bench import statistics, verify_bundle


def validate_schedule(policy, rows):
    if policy not in ("full", "periodic") or not rows:
        raise ValueError("Require known policy and nonempty rows")
    scheduler, starts = SearchSchedule(), []
    for row in rows:
        expected = scheduler.begin(row["start_ns"], force_full=policy == "full")
        if any(row[k] != v for k, v in expected.items()):
            raise ValueError("Recorded schedule differs from timestamp-replayed decisions")
        scheduler.finish(row["end_ns"])
        duration = (row["end_ns"] - row["start_ns"]) / 1e6
        if duration != row["scheduled_processing_ms"] or duration < row["processing_ms"]:
            raise ValueError("Scheduled duration does not match timestamp evidence")
        for key in (
            "scheduled_processing_ms",
            "processing_ms",
            "search_ms",
            "inference_ms",
            "selection_ms",
        ):
            statistics([row[key]])
        if row["mode"] == "mser_direct":
            starts.append(row["start_ns"])
    return dict(
        full_start_intervals_ms=[(b - a) / 1e6 for a, b in zip(starts, starts[1:])],
        max_dispatch_lateness_ms=max(r["due_lateness_ns"] for r in rows) / 1e6,
        mode_counts={k: sum(r["mode"] == k for r in rows) for k in ("mser_direct", "mser_bright")},
    )


def review(base, results):
    base, results = Path(base), Path(results)
    verify_bundle(base)
    inputs, goldens = read_json(base / "inputs.json"), read_json(base / "search-golden.json")
    groups, runs, identity = {"full": [], "periodic": []}, {}, None
    for path in sorted(results.glob("*.json")):
        report, log = read_json(path), path.with_suffix(".frames.jsonl")
        policy = report["policy"]
        rows = [json.loads(s) for s in log.read_text().splitlines()]
        n = len(inputs["frames"])
        if (
            policy not in groups
            or report["machine"] != "armv6l"
            or report["camera_used"] is not False
            or report["flight_commands"] is not None
            or report["test_evaluated"] is not False
            or report["deployment_approved"] is not False
            or report["temporal_recall_measured"] is not False
            or report["hard_reacquisition_deadline_proven"] is not False
            or type(report["rounds"]) is not int
            or not 1 <= report["rounds"] <= 10
            or not n
            or report["frames"] != len(rows)
            or report["unique_frames"] != n
            or len(rows) != report["rounds"] * n
            or report["interval_ns"] != SearchSchedule.interval_ns
            or report["max_bright_frames"] != SearchSchedule.max_bright_frames
        ):
            raise ValueError("Require complete bounded ARMv6 research replay")
        for key, expected in dict(
            frames_log_sha256=sha256(log),
            bundle_sha256=sha256(base / "bundle.json"),
            inputs_sha256=sha256(base / "inputs.json"),
            golden_sha256=sha256(base / "search-golden.json"),
            model_sha256=sha256(base / inputs["models"][report["model"]]["path"]),
            script_sha256=sha256(base / "pi_balloon_schedule_bench.py"),
            scheduler_sha256=sha256(base / "dtr/research_search_schedule.py"),
        ).items():
            if report[key] != expected:
                raise ValueError("Report artifact identity changed")
        current = tuple(
            report[k]
            for k in (
                "model_sha256",
                "bundle_sha256",
                "runtime_library_sha256",
                "region_library_sha256",
                "opencv",
                "script_sha256",
                "scheduler_sha256",
            )
        )
        if not all(current) or (identity is not None and current != identity):
            raise ValueError("Trials are not a matched hardware comparison")
        identity = current
        validated = validate_schedule(policy, rows)
        if any(report[k] != v for k, v in validated.items()):
            raise ValueError("Schedule summary differs from raw evidence")
        frames = []
        golden = goldens[report["model"]]
        for position, row in enumerate(rows):
            repeat, index = divmod(position, n)
            source = inputs["frames"][index]
            if (
                row["round"] != repeat
                or row["index"] != index
                or row["source"] != source["source"]
                or row["panel"] != source["panel"]
            ):
                raise ValueError("Frame ordering/source mismatch")
            key = "mser" if row["mode"] == "mser_direct" else "mser_bright"
            check_predictions(row["detections"], golden[key][index]["detections"])
            counts = detection_counts(source["truth"], row["detections"])
            full_counts = detection_counts(source["truth"], golden["mser"][index]["detections"])
            frames.append(
                dict(
                    round=repeat,
                    index=index,
                    mode=row["mode"],
                    counts=counts,
                    same_counts_as_full=counts == full_counts,
                )
            )
        runs[path.stem] = dict(
            report=report,
            report_sha256=sha256(path),
            validated_schedule=validated,
            frame_checks=frames,
        )
        groups[policy].append((path.stem, rows))
    if any(len(g) < 2 for g in groups.values()):
        raise ValueError("Require at least two trials per policy")
    pooled = {}
    for policy, trials in groups.items():
        rows = [r for _, records in trials for r in records]
        timing = statistics([r["scheduled_processing_ms"] for r in rows])
        intervals = [
            v
            for name, _ in trials
            for v in runs[name]["validated_schedule"]["full_start_intervals_ms"]
        ]
        completion_intervals = []
        for _, records in trials:
            ends = [r["end_ns"] for r in records if r["mode"] == "mser_direct"]
            completion_intervals.extend((b - a) / 1e6 for a, b in zip(ends, ends[1:]))
        pooled[policy] = dict(
            trials=[name for name, _ in trials],
            timing_samples=len(rows),
            timing=timing,
            processing_fps=1000 / timing["mean_ms"],
            mode_counts={
                k: sum(r["mode"] == k for r in rows) for k in ("mser_direct", "mser_bright")
            },
            full_start_intervals=statistics(intervals),
            full_completion_intervals=statistics(completion_intervals),
            mean_neural_calls=sum(r["neural_calls"] for r in rows) / len(rows),
            max_dispatch_lateness_ms=max(r["due_lateness_ns"] for r in rows) / 1e6,
        )
    return dict(
        runs=runs,
        pooled=pooled,
        processing_speedup=pooled["periodic"]["processing_fps"] / pooled["full"]["processing_fps"],
        all_timed_counts_match_full=all(
            f["same_counts_as_full"] for r in runs.values() for f in r["frame_checks"]
        ),
        scope="Clock-replayed scheduler evidence on arbitrary photos; not temporal target recall",
        reviewer_sha256=sha256(__file__),
        camera_used=False,
        test_evaluated=False,
        temporal_recall_measured=False,
        hard_reacquisition_deadline_proven=False,
        deployment_approved=False,
        flight_qualified=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "results", "output"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    report = review(args.base, args.results)
    write_json(args.output, report)
    print(report["pooled"])
    print(
        dict(
            speedup=report["processing_speedup"], counts_match=report["all_timed_counts_match_full"]
        )
    )
