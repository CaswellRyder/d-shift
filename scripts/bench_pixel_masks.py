"""Research-only: time exact RGB-distance mask variants and the component pass on bundle frames.

Every variant must reproduce the NumPy lookup-table masks bit for bit; timing is reported per
stage so the proposal search cost can be attributed. Development frames only; no camera.
"""
import argparse
import platform
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, write_json

try:
    from research_pixel_proposals import MASKS, color, pixel_search
except ModuleNotFoundError:
    from scripts.research_pixel_proposals import MASKS, color, pixel_search


def stats(values):
    return dict(mean_ms=float(np.mean(values)), p95_ms=float(np.percentile(values, 95)))


def timed(fn, images, repeats):
    for rgb in images:
        fn(rgb)
    times = []
    for _ in range(repeats):
        for rgb in images:
            start = time.perf_counter()
            fn(rgb)
            times.append((time.perf_counter() - start) * 1000)
    return stats(times)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--red", type=color, required=True)
    parser.add_argument("--blue", type=color, required=True)
    parser.add_argument("--threshold", type=float, default=75)
    parser.add_argument("--min-area", type=int, default=53)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    root = Path(args.base)
    inputs = read_json(root / "inputs.json")
    images = [np.asarray(Image.open(root / f["path"]).convert("RGB")) for f in inputs["frames"]]
    cv2.setNumThreads(1)
    colors = (args.red, args.blue)
    variants = {name: cls(colors, args.threshold) for name, cls in MASKS.items()}
    reference = [variants["numpy"](rgb) for rgb in images]
    for name, masks in variants.items():
        for rgb, expected in zip(images, reference):
            for got, want in zip(masks(rgb), expected):
                if not np.array_equal(got != 0, want != 0):
                    raise RuntimeError(f"{name} mask differs from the NumPy reference")
        searched = [pixel_search(rgb, masks, args.min_area) for rgb in images]
        if searched != [pixel_search(rgb, variants["numpy"], args.min_area) for rgb in images]:
            raise RuntimeError(f"{name} proposals differ from the NumPy reference")
    result = dict(purpose="Exact pixel-mask timing on development frames; NOT qualification",
                  deployment_approved=False, test_evaluated=False, threshold=args.threshold,
                  inputs_sha256=sha256(root / "inputs.json"), script_sha256=sha256(__file__),
                  machine=platform.machine(), opencv=cv2.__version__, numpy=np.__version__,
                  repeats=args.repeats, masks={}, search={})
    for name, masks in variants.items():
        result["masks"][name] = timed(masks, images, args.repeats)
        result["search"][name] = timed(lambda rgb, m=masks: pixel_search(rgb, m, args.min_area),
                                       images, args.repeats)
        print(f"{name}: mask {result['masks'][name]['mean_ms']:.1f} ms, "
              f"search {result['search'][name]['mean_ms']:.1f} ms", flush=True)
    times = []
    for _ in range(args.repeats):
        for masks in reference:
            start = time.perf_counter()
            for mask in masks:
                cv2.connectedComponentsWithStats(mask, connectivity=8)
            times.append((time.perf_counter() - start) * 1000)
    result["components"] = stats(times)
    print(f"components (both colors): {result['components']['mean_ms']:.1f} ms")
    write_json(args.output, result)


if __name__ == "__main__":
    main()
