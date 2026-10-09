"""Isolated periodic-search replay; arbitrary photos are NOT a motion sequence."""

import argparse
import json
import os
from pathlib import Path
import platform
import resource
import time

import cv2
import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, write_json
from dtr.native_balloon_regions import NativeBalloonRegions
from dtr.research_search_schedule import SearchSchedule
from dtr.runtime import Predictor

try:
    from pi_balloon_search_bench import check_predictions, process_frame
    from pi_red_blue_bench import health, statistics, verify_bundle
except ModuleNotFoundError:
    from scripts.pi_balloon_search_bench import check_predictions, process_frame
    from scripts.pi_red_blue_bench import health, statistics, verify_bundle


def scheduled_frame(rgb, predictor, native, scheduler, policy, clock_ns=None):
    if policy not in ("full", "periodic"):
        raise ValueError("Unknown search policy")
    clock_ns = clock_ns or time.monotonic_ns
    start = clock_ns()
    decision = scheduler.begin(start, force_full=policy == "full")
    try:
        observed = process_frame(rgb, predictor, decision["mode"], native)
    except BaseException:
        scheduler.finish(clock_ns(), success=False)
        raise
    end = clock_ns()
    scheduler.finish(end)
    return dict(
        **observed,
        **decision,
        start_ns=start,
        end_ns=end,
        scheduled_processing_ms=(end - start) / 1e6,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "region-library", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--policy", choices=("full", "periodic"), required=True)
    parser.add_argument("--model", default="new-views-42")
    parser.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args()
    root, output = Path(args.base).resolve(), Path(args.output)
    log = output.with_suffix(".frames.jsonl")
    if output.exists() or log.exists():
        raise FileExistsError(output)
    if not 1 <= args.rounds <= 10:
        parser.error("Require 1..10 complete replay rounds")
    verify_bundle(root)
    native = NativeBalloonRegions(args.region_library, direct=True)
    receipt = read_json(native.path.with_suffix(".json"))
    if receipt["source_sha256"] != sha256(root / "dtr/native_balloon_regions.c"):
        raise ValueError("Native reducer source mismatch")
    for key in ("DTR_RED_BLUE_COLOR_LOOKUP", "DTR_RED_BLUE_LOOKUP_LIBRARY"):
        os.environ.pop(key, None)
    cv2.setNumThreads(1)
    inputs = read_json(root / "inputs.json")
    golden = read_json(root / "search-golden.json")[args.model]
    records = inputs["frames"]
    if not records or any(len(golden[k]) != len(records) for k in ("mser", "mser_bright")):
        raise ValueError("Incomplete policy goldens")
    model = root / inputs["models"][args.model]["path"]
    predictor = Predictor(model, allow_unvalidated=True)
    images = [np.asarray(Image.open(root / r["path"]).convert("RGB")) for r in records]
    before, max_delta = health(), 0.0
    # Same warmup for both policies; all input pixels/model loading precede timing.
    for mode in ("mser_direct", "mser_bright"):
        key = "mser" if mode == "mser_direct" else mode
        for rgb, expected in zip(images, golden[key], strict=True):
            observed = process_frame(rgb, predictor, mode, native)
            max_delta = max(
                max_delta, check_predictions(observed["detections"], expected["detections"])
            )
    scheduler, rows = SearchSchedule(), []
    cpu_start, started = time.process_time(), time.perf_counter()
    for repeat in range(args.rounds):
        for index, (rgb, frame) in enumerate(zip(images, records, strict=True)):
            observed = scheduled_frame(rgb, predictor, native, scheduler, args.policy)
            rows.append(
                dict(
                    round=repeat,
                    index=index,
                    source=frame["source"],
                    panel=frame["panel"],
                    **observed,
                )
            )
    elapsed, cpu = time.perf_counter() - started, time.process_time() - cpu_start
    after = health()
    # Golden comparisons and disk writes do not influence scheduler dispatch timing.
    for row in rows:
        key = "mser" if row["mode"] == "mser_direct" else row["mode"]
        max_delta = max(
            max_delta, check_predictions(row["detections"], golden[key][row["index"]]["detections"])
        )
    starts = [r["start_ns"] for r in rows if r["mode"] == "mser_direct"]
    report = dict(
        policy=args.policy,
        model=args.model,
        machine=platform.machine(),
        opencv=cv2.__version__,
        model_sha256=sha256(model),
        bundle_sha256=sha256(root / "bundle.json"),
        inputs_sha256=sha256(root / "inputs.json"),
        golden_sha256=sha256(root / "search-golden.json"),
        script_sha256=sha256(__file__),
        scheduler_sha256=sha256(root / "dtr/research_search_schedule.py"),
        runtime_library_sha256=sha256(os.environ["DTR_TFLITE_LIBRARY"])
        if os.environ.get("DTR_TFLITE_LIBRARY")
        else None,
        region_library_sha256=sha256(native.path),
        region_build_receipt=receipt,
        interval_ns=SearchSchedule.interval_ns,
        max_bright_frames=SearchSchedule.max_bright_frames,
        frames=len(rows),
        unique_frames=len(records),
        rounds=args.rounds,
        mode_counts={k: sum(r["mode"] == k for r in rows) for k in ("mser_direct", "mser_bright")},
        timing={
            k: statistics([r[k] for r in rows])
            for k in (
                "processing_ms",
                "scheduled_processing_ms",
                "search_ms",
                "inference_ms",
                "selection_ms",
            )
        },
        full_start_intervals_ms=[(b - a) / 1e6 for a, b in zip(starts, starts[1:])],
        max_dispatch_lateness_ms=max(r["due_lateness_ns"] for r in rows) / 1e6,
        mean_neural_calls=float(np.mean([r["neural_calls"] for r in rows])),
        timed_loop_wall_seconds=elapsed,
        timed_loop_cpu_seconds=cpu,
        health_before=before,
        health_after=after,
        max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        max_host_pi_score_delta=max_delta,
        camera_used=False,
        flight_commands=None,
        test_evaluated=False,
        deployment_approved=False,
        temporal_recall_measured=False,
        hard_reacquisition_deadline_proven=False,
        scope="Timing over arbitrary ordered development photos, not a representative motion sequence",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with log.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    report["frames_log_sha256"] = sha256(log)
    write_json(output, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
