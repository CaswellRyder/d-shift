"""Isolated exact-mask/search comparison; identical thresholds, no inference or actuation."""
import argparse
import json
import os
from pathlib import Path
import platform
import time

import cv2
import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, write_json
from dtr.vision import RED_BLUE_PROFILE, color_masks, proposals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--native-library", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not 1 <= args.rounds <= 50:
        parser.error("rounds must be 1..50")
    root = args.base.resolve()
    receipt = read_json(root / "bundle.json")
    for relative, digest in receipt["files"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or sha256(path) != digest:
            raise ValueError("Bundle changed or escaped root")
    cv2.setNumThreads(1)
    inputs = read_json(root / "inputs.json")
    frames = [np.asarray(Image.open(root / r["path"]).convert("RGB")) for r in inputs["frames"]]
    backends = ("hsv", "lookup", "native") if args.native_library else ("hsv", "lookup")
    timings = {backend: {stage: [] for stage in ("masks", "proposals")} for backend in backends}
    references = []
    os.environ.pop("DTR_RED_BLUE_COLOR_LOOKUP", None)
    os.environ.pop("DTR_RED_BLUE_LOOKUP_LIBRARY", None)
    for rgb in frames:
        references.append(proposals(rgb, "balloon", profile=RED_BLUE_PROFILE))
    comparisons = 0
    # Warm both implementations, then alternate order; neither runs concurrently.
    for iteration in range(args.rounds+1):
        for backend in (backends if iteration % 2 == 0 else backends[::-1]):
            os.environ.pop("DTR_RED_BLUE_COLOR_LOOKUP", None)
            os.environ.pop("DTR_RED_BLUE_LOOKUP_LIBRARY", None)
            if backend != "hsv":
                os.environ["DTR_RED_BLUE_COLOR_LOOKUP"] = str(root / "red-blue-lookup.bin")
            if backend == "native":
                os.environ["DTR_RED_BLUE_LOOKUP_LIBRARY"] = str(args.native_library.resolve())
            for rgb, expected in zip(frames, references):
                start = time.perf_counter()
                color_masks(rgb, "balloon", RED_BLUE_PROFILE)
                masks_done = time.perf_counter()
                found = proposals(rgb, "balloon", profile=RED_BLUE_PROFILE)
                done = time.perf_counter()
                if found != expected:
                    raise ValueError("Proposal order/geometry/score parity failed")
                comparisons += 1
                if iteration:
                    timings[backend]["masks"].append((masks_done-start)*1000)
                    timings[backend]["proposals"].append((done-masks_done)*1000)
    os.environ.pop("DTR_RED_BLUE_COLOR_LOOKUP", None)
    os.environ.pop("DTR_RED_BLUE_LOOKUP_LIBRARY", None)
    summary = {backend: {stage: dict(mean_ms=float(np.mean(values)), p95_ms=float(np.percentile(values, 95)))
                        for stage, values in rows.items()} for backend, rows in timings.items()}
    report = dict(host=platform.node(), machine=platform.machine(), opencv=cv2.__version__,
                  bundle_sha256=sha256(root / "bundle.json"), comparisons_passed=comparisons,
                  unique_photos=len(frames), rounds=args.rounds, timings_ms=timings, summary=summary,
                  reference_proposals=references, deployment_approved=False, test_evaluated=False,
                  native_library_sha256=sha256(args.native_library) if args.native_library else None,
                  scope="Warmed search-only development timing; not camera FPS or accuracy")
    write_json(args.output, report)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
