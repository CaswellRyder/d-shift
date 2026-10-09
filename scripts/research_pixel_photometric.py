"""Research-only: do fixed-color pixel proposals survive exposure and white-balance shifts?

Applies fixed synthetic photometric transforms (chosen before scoring) to the development frames
and compares MSER search with pixel-mask proposals, both followed by the unchanged classifier.
Desktop replay of development frames only; synthetic shifts are not real lighting; NOT
qualification.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, write_json
from dtr.runtime import Predictor
from dtr.vision import suppress_duplicates

try:
    from research_balloon_search import experimental, suppress_confirmed_balloon_parts
    from research_pixel_proposals import CvPixelMasks, color, coverage, pixel_search
except ModuleNotFoundError:
    from scripts.research_balloon_search import experimental, suppress_confirmed_balloon_parts
    from scripts.research_pixel_proposals import CvPixelMasks, color, coverage, pixel_search

# (name, per-channel RGB gain, gamma); applied as clip(255 * (gain * x / 255) ** gamma).
TRANSFORMS = (
    ("identity", (1., 1., 1.), 1.),
    ("darker_0.6", (.6, .6, .6), 1.),
    ("darker_0.8", (.8, .8, .8), 1.),
    ("brighter_1.25", (1.25, 1.25, 1.25), 1.),
    ("gamma_0.7", (1., 1., 1.), .7),
    ("gamma_1.4", (1., 1., 1.), 1.4),
    ("warm_cast", (1.15, 1., .85), 1.),
    ("cool_cast", (.85, 1., 1.15), 1.),
)


def photometric(rgb, gain, gamma):
    lut = np.empty((256, 1, 3), np.uint8)
    v = np.arange(256, dtype=np.float64)
    for channel, g in enumerate(gain):
        lut[:, 0, channel] = np.clip(np.rint(255 * np.clip(g * v / 255, 0, None) ** gamma), 0, 255)
    return cv2.LUT(rgb, lut)


def classify(rgb, proposals, predictor):
    found = []
    for proposal in proposals:
        a, b, c, d = proposal["crop_box"]
        pred = predictor.predict(rgb[b:d, a:c])
        found.append(dict(proposal, **{k: pred[k] for k in ("label", "scores", "score", "accepted")}))
    return suppress_confirmed_balloon_parts(suppress_duplicates(found))


def main():
    from scripts.review_pixel_balloon_bench import panel_counts

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--model", required=True, help="bundle model key, or a .tflite path")
    parser.add_argument("--searches", choices=("all", "mser"), default="all")
    parser.add_argument("--red", type=color, required=True)
    parser.add_argument("--blue", type=color, required=True)
    parser.add_argument("--threshold", type=float, action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    root = Path(args.base)
    inputs = read_json(root / "inputs.json")
    frames = inputs["frames"]
    images = [np.asarray(Image.open(root / f["path"]).convert("RGB")) for f in frames]
    cv2.setNumThreads(1)
    model = (Path(args.model) if args.model.endswith(".tflite")
             else root / inputs["models"][args.model]["path"])
    predictor = Predictor(model, allow_unvalidated=True)
    searches = {"mser": lambda rgb: experimental(rgb, "mser")}
    for threshold in args.threshold if args.searches == "all" else ():
        masks = CvPixelMasks((args.red, args.blue), threshold)
        for stride in (1, 2):
            searches[f"pixel_t{threshold:g}_s{stride}"] = (
                lambda rgb, m=masks, s=stride: pixel_search(rgb, m, 53, s))
    rows = []
    for name, gain, gamma in TRANSFORMS:
        shifted = [photometric(rgb, gain, gamma) for rgb in images]
        for search, fn in searches.items():
            proposals = [fn(rgb) for rgb in shifted]
            detections = [classify(rgb, p, predictor) for rgb, p in zip(shifted, proposals)]
            counts = panel_counts(frames, detections)
            total = {k: sum(c[k] for by in counts.values() for c in by.values())
                     for k in ("tp", "fp", "fn")}
            covered = sum(coverage(f["truth"], [p["box"] for p in ps], [p["color_group"] for p in ps])
                          for f, ps in zip(frames, proposals))
            calls = sum(len(p) for p in proposals) / len(frames)
            rows.append(dict(transform=name, gain=list(gain), gamma=gamma, search=search,
                             coverage=covered, calls_per_frame=calls, counts=counts, total=total))
            print(f"{name:14s} {search:16s} cover {covered:2d}/28 calls {calls:4.2f} "
                  f"TP/FP/FN {total['tp']}/{total['fp']}/{total['fn']}", flush=True)
    write_json(args.output, dict(
        purpose="Synthetic photometric robustness of proposals; development frames, NOT "
                "qualification; synthetic shifts are not real lighting",
        deployment_approved=False, test_evaluated=False, model=args.model, model_sha256=sha256(model),
        colors=dict(red_balloon=list(args.red), blue_balloon=list(args.blue)),
        inputs_sha256=sha256(root / "inputs.json"), script_sha256=sha256(__file__),
        rows=rows))


if __name__ == "__main__":
    main()
