"""Research-only: split per-crop classifier time into preprocessing, invoke and postprocessing.

Crops are the stride-2 OpenCV pixel proposals on the bundle frames, so the size mix matches
the proposal pipeline. Development frames only; no camera; NOT qualification.
"""
import argparse
import platform
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, write_json
from dtr.runtime import Predictor, softmax

try:
    from research_pixel_proposals import CvPixelMasks, color, pixel_search
except ModuleNotFoundError:
    from scripts.research_pixel_proposals import CvPixelMasks, color, pixel_search


def stage_times(predictor, crop):
    size = predictor.metadata["size"]
    t0 = time.perf_counter()
    values = np.asarray(Image.fromarray(crop).resize((size, size), Image.Resampling.BILINEAR),
                        dtype=np.float32)[None]
    t1 = time.perf_counter()
    predictor.interpreter.set_tensor(predictor.input["index"], values)
    predictor.interpreter.invoke()
    t2 = time.perf_counter()
    softmax(predictor.interpreter.get_tensor(predictor.output["index"])[0].astype(np.float32))
    t3 = time.perf_counter()
    resized = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR).astype(np.float32)[None]
    t4 = time.perf_counter()
    return dict(pil_preprocess=t1 - t0, set_and_invoke=t2 - t1, postprocess=t3 - t2,
                cv2_preprocess=t4 - t3), values, resized


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--red", type=color, required=True)
    parser.add_argument("--blue", type=color, required=True)
    parser.add_argument("--threshold", type=float, default=75)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    root = Path(args.base)
    inputs = read_json(root / "inputs.json")
    cv2.setNumThreads(1)
    predictor = Predictor(root / inputs["models"][args.model]["path"], allow_unvalidated=True)
    masks = CvPixelMasks((args.red, args.blue), args.threshold)
    crops = []
    for frame in inputs["frames"]:
        rgb = np.asarray(Image.open(root / frame["path"]).convert("RGB"))
        for p in pixel_search(rgb, masks, 53, 2):
            a, b, c, d = p["crop_box"]
            crops.append(np.ascontiguousarray(rgb[b:d, a:c]))
    for crop in crops:  # warm-up
        stage_times(predictor, crop)
    totals, pixel_delta = {}, 0.
    for _ in range(args.repeats):
        for crop in crops:
            times, pil, cv = stage_times(predictor, crop)
            pixel_delta = max(pixel_delta, float(np.abs(pil - cv).max()))
            for k, v in times.items():
                totals.setdefault(k, []).append(v * 1000)
    result = dict(purpose="Per-crop classifier stage timing; development frames, NOT qualification",
                  deployment_approved=False, test_evaluated=False, model=args.model,
                  crops=len(crops), repeats=args.repeats, machine=platform.machine(),
                  script_sha256=sha256(__file__), inputs_sha256=sha256(root / "inputs.json"),
                  crop_sizes=[list(c.shape[:2]) for c in crops],
                  max_pixel_delta_cv2_vs_pil=pixel_delta,
                  stages_ms={k: dict(mean=float(np.mean(v)), p95=float(np.percentile(v, 95)))
                             for k, v in totals.items()})
    write_json(args.output, result)
    for k, v in result["stages_ms"].items():
        print(f"{k}: {v['mean']:.2f} ms mean, {v['p95']:.2f} p95")
    print(f"{len(crops)} crops; max |cv2 - PIL| pixel {pixel_delta:.1f}")


if __name__ == "__main__":
    main()
