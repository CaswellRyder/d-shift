"""Verify live periodic-search trials from raw timestamps; not recall or a deadline."""

import argparse
import json
from pathlib import Path

from dtr.data import read_json, sha256, write_json
from dtr.research_search_schedule import SearchSchedule
from scripts.pi_balloon_live_schedule_bench import trial_summary
from scripts.pi_red_blue_bench import statistics, verify_bundle
from scripts.review_balloon_live import review as review_live
from scripts.review_scheduled_balloon_search import validate_schedule

SCRIPTS = Path(__file__).resolve().parent


def review(base, results):
    base = Path(base)
    verify_bundle(base)
    inputs = read_json(base / "inputs.json")
    expected_identity = dict(
        bundle_sha256=sha256(base / "bundle.json"),
        scheduler_sha256=sha256(base / "dtr/research_search_schedule.py"),
        script_sha256=sha256(SCRIPTS / "pi_balloon_live_schedule_bench.py"),
        live_helper_sha256=sha256(SCRIPTS / "pi_balloon_live_bench.py"),
    )
    groups, runs, identity = {"full": [], "periodic": []}, {}, None
    for root in sorted(p for p in Path(results).iterdir() if (p / "report.json").is_file()):
        live = review_live(root)
        report = live["report"]
        rows = [json.loads(s) for s in (root / "frames.jsonl").read_text().splitlines()]
        policy = report.get("policy")
        if (
            policy not in groups
            or report["temporal_recall_measured"] is not False
            or report["hard_reacquisition_deadline_proven"] is not False
            or report["sensor_clock"] != "CLOCK_BOOTTIME"
            or report["scheduler_clock"] != "CLOCK_BOOTTIME"
            or report["interval_ns"] != SearchSchedule.interval_ns
            or report["max_bright_frames"] != SearchSchedule.max_bright_frames
            or report["model_sha256"] != sha256(base / inputs["models"][report["model"]]["path"])
            or any(report[k] != v for k, v in expected_identity.items())
        ):
            raise ValueError("Require bounded ARMv6 live scheduler research from known files")
        current = tuple(
            report[k]
            for k in (
                "model_sha256",
                "runtime_library_sha256",
                "region_library_sha256",
                "opencv",
                "requested_camera_fps",
            )
        )
        if not all(current) or (identity is not None and current != identity):
            raise ValueError("Trials are not a matched hardware comparison")
        identity = current
        validated = validate_schedule(policy, rows)
        fresh = trial_summary(rows, report["elapsed_s"])
        if any(report[k] != v for k, v in {**validated, **fresh}.items()):
            raise ValueError("Schedule or freshness summary differs from raw evidence")
        runs[root.name] = dict(
            live_review={k: v for k, v in live.items() if k != "report"},
            report=report,
            validated_schedule=validated,
        )
        groups[policy].append((root.name, rows, fresh))
    if any(len(g) < 2 for g in groups.values()):
        raise ValueError("Require at least two trials per policy")
    pooled = {}
    for policy, trials in groups.items():
        rows = [r for _, records, _ in trials for r in records]
        ages = [a for _, _, f in trials for a in f["full_observation_age_ms"] if a is not None]
        elapsed = sum(runs[name]["report"]["elapsed_s"] for name, _, _ in trials)

        def pool(key, trials=trials):
            values = [v for _, _, f in trials for v in f[key]]
            return statistics(values) if values else None

        pooled[policy] = dict(
            trials=[name for name, _, _ in trials],
            frames=len(rows),
            live_fps=len(rows) / elapsed,
            mode_counts={
                k: sum(r["mode"] == k for r in rows) for k in ("mser_direct", "mser_bright")
            },
            mean_neural_calls=sum(r["neural_calls"] for r in rows) / len(rows),
            scheduled_processing=statistics([r["scheduled_processing_ms"] for r in rows]),
            frame_wall=statistics([r["frame_wall_ms"] for r in rows]),
            sensor_to_result=statistics(
                [r["sensor_to_result_ms"] for r in rows if r["timestamp_valid"]]
            ),
            full_observation_age=statistics(ages),
            unknown_full_observation_age_frames=sum(
                f["unknown_full_observation_age_frames"] for _, _, f in trials
            ),
            full_result_intervals=pool("full_result_intervals_ms"),
            full_sensor_intervals=pool("full_sensor_intervals_ms"),
            max_dispatch_lateness_ms=max(r["due_lateness_ns"] for r in rows) / 1e6,
            accepted_detections=sum(d["accepted"] for r in rows for d in r["detections"]),
            nearly_black_samples=sum(
                runs[name]["live_review"]["nearly_black_samples"] for name, _, _ in trials
            ),
        )
    return dict(
        runs=runs,
        pooled=pooled,
        live_fps_ratio=pooled["periodic"]["live_fps"] / pooled["full"]["live_fps"],
        scope="Unlabeled live scene; scheduler timing and observation age only",
        reviewer_sha256=sha256(__file__),
        camera_used=True,
        accuracy_measured=False,
        temporal_recall_measured=False,
        hard_reacquisition_deadline_proven=False,
        test_evaluated=False,
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
    result = review(args.base, args.results)
    write_json(args.output, result)
    print(json.dumps(result["pooled"], indent=2))
    print(dict(live_fps_ratio=result["live_fps_ratio"]))
