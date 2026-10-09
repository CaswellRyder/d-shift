"""Sequential current-student replay benchmark; no camera, training or actuation."""

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
from dtr.runtime import Predictor
from dtr.vision import suppress_duplicates

try:
    from research_balloon_search import experimental, suppress_confirmed_balloon_parts
    from pi_red_blue_bench import health, statistics, verify_bundle
except ModuleNotFoundError:
    from scripts.research_balloon_search import experimental, suppress_confirmed_balloon_parts
    from scripts.pi_red_blue_bench import health, statistics, verify_bundle


def process_frame(rgb, predictor, search):
    if search not in ("baseline", "mser"):
        raise ValueError("Unsupported search")
    start = time.perf_counter()
    proposals = experimental(rgb, search)
    searched = time.perf_counter()
    found = []
    for proposal in proposals:
        a, b, c, d = proposal["crop_box"]
        pred = predictor.predict(rgb[b:d, a:c])
        found.append(
            dict(proposal, **{k: pred[k] for k in ("label", "scores", "score", "accepted")})
        )
    inferred = time.perf_counter()
    detections = suppress_confirmed_balloon_parts(suppress_duplicates(found))
    done = time.perf_counter()
    return dict(
        detections=detections,
        processing_ms=(done - start) * 1000,
        search_ms=(searched - start) * 1000,
        inference_ms=(inferred - searched) * 1000,
        selection_ms=(done - inferred) * 1000,
        neural_calls=len(found),
    )


def check_predictions(actual, expected, tolerance=0.001):
    if len(actual) != len(expected):
        raise ValueError("Host/Pi proposal count mismatch")
    max_delta = 0.0
    for a, b in zip(actual, expected):
        if any(
            a[k] != b[k]
            for k in ("box", "crop_box", "label", "accepted", "raw_accepted", "suppressed")
        ):
            raise ValueError("Host/Pi geometry, classification or selection mismatch")
        scores, reference = np.asarray(a["scores"]), np.asarray(b["scores"])
        if (
            scores.shape != (3,)
            or reference.shape != (3,)
            or not np.isfinite(scores).all()
            or not np.isfinite(reference).all()
        ):
            raise ValueError("Invalid golden scores")
        delta = float(np.abs(scores - reference).max())
        if delta > tolerance:
            raise ValueError("Host/Pi score tolerance exceeded")
        max_delta = max(max_delta, delta)
    return max_delta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--search", choices=("baseline", "mser"), required=True)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not 1 <= args.rounds <= 10:
        parser.error("Require 1..10 complete replay rounds")
    root, out = Path(args.base).resolve(), Path(args.output)
    log = out.with_suffix(".frames.jsonl")
    if out.exists() or log.exists():
        raise FileExistsError(out)
    verify_bundle(root)
    for key in ("DTR_RED_BLUE_COLOR_LOOKUP", "DTR_RED_BLUE_LOOKUP_LIBRARY"):
        os.environ.pop(key, None)
    cv2.setNumThreads(1)
    inputs = read_json(root / "inputs.json")
    golden = read_json(root / "search-golden.json")[args.model][args.search]
    records = inputs["frames"]
    if not records or len(golden) != len(records):
        raise ValueError("Golden frame coverage mismatch")
    model = root / inputs["models"][args.model]["path"]
    predictor = Predictor(model, allow_unvalidated=True)
    images = [np.asarray(Image.open(root / r["path"]).convert("RGB")) for r in records]
    max_delta = 0.0
    before = health()
    for rgb, expected in zip(images, golden):
        observed = process_frame(rgb, predictor, args.search)
        max_delta = max(
            max_delta, check_predictions(observed["detections"], expected["detections"])
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    cpu_start = time.process_time()
    started = time.perf_counter()
    for repeat in range(args.rounds):
        for index, (rgb, frame) in enumerate(zip(images, records)):
            observed = process_frame(rgb, predictor, args.search)
            max_delta = max(
                max_delta, check_predictions(observed["detections"], golden[index]["detections"])
            )
            rows.append(
                dict(
                    round=repeat,
                    index=index,
                    source=frame["source"],
                    panel=frame["panel"],
                    **observed,
                )
            )
    elapsed = time.perf_counter() - started
    cpu = time.process_time() - cpu_start
    report = dict(
        model=args.model,
        search=args.search,
        machine=platform.machine(),
        host=platform.node(),
        opencv=cv2.__version__,
        runtime=type(predictor.interpreter).__module__,
        runtime_library_sha256=sha256(os.environ["DTR_TFLITE_LIBRARY"])
        if os.environ.get("DTR_TFLITE_LIBRARY")
        else None,
        model_sha256=sha256(model),
        bundle_sha256=sha256(root / "bundle.json"),
        inputs_sha256=sha256(root / "inputs.json"),
        golden_sha256=sha256(root / "search-golden.json"),
        script_sha256=sha256(__file__),
        frames=len(rows),
        unique_frames=len(records),
        rounds=args.rounds,
        max_host_pi_score_delta=max_delta,
        timing={
            key: statistics([r[key] for r in rows])
            for key in ("processing_ms", "search_ms", "inference_ms", "selection_ms")
        },
        mean_neural_calls=float(np.mean([r["neural_calls"] for r in rows])),
        timed_loop_wall_seconds=elapsed,
        timed_loop_cpu_seconds=cpu,
        cpu_wall_ratio=cpu / elapsed,
        health_before=before,
        health_after=health(),
        max_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        rss_units="KiB on Linux; bytes on macOS",
        camera_used=False,
        flight_commands=None,
        test_evaluated=False,
        deployment_approved=False,
        scope="Warmed 16-development-scene replay, camera excluded; repetitions are timing samples, not independent accuracy",
    )
    with log.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    report["frames_log_sha256"] = sha256(log)
    write_json(out, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
