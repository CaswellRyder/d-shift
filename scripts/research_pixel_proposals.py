"""Research-only: RGB-distance pixel masks as balloon proposals for the neural classifier.

The mask is the other team's rule (squared RGB distance to a reference color <= threshold^2).
Components replace MSER regions; candidate geometry, selection, classification and
suppression are unchanged. Development frames only; NOT flight qualification.
"""
import argparse
import platform
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, write_json
from dtr.runtime import Predictor
from dtr.vision import suppress_duplicates

try:
    from research_balloon_search import select, suppress_confirmed_balloon_parts
except ModuleNotFoundError:
    from scripts.research_balloon_search import select, suppress_confirmed_balloon_parts

GROUPS = ("red_balloon", "blue_balloon")


class PixelMasks:
    """Exact squared-distance masks via per-channel lookup tables, one per color group."""

    def __init__(self, colors, threshold):
        v = np.arange(256, dtype=np.int32)
        self.luts = [[((v - int(t)) ** 2).astype(np.int32) for t in color] for color in colors]
        self.thr_sq = int(float(threshold) ** 2)

    def __call__(self, rgb):
        planes = [rgb[:, :, i] for i in range(3)]
        masks = []
        for luts in self.luts:
            acc = np.take(luts[0], planes[0])
            acc += np.take(luts[1], planes[1])
            acc += np.take(luts[2], planes[2])
            masks.append((acc <= self.thr_sq).astype(np.uint8))
        return masks


class CvPixelMasks:
    """The same masks with OpenCV: one 3-channel uint16 LUT per color, channel sum, compare.

    Per-channel squares are capped at threshold^2 + 1, which changes no decision (that channel
    alone already exceeds the threshold) and keeps the three-channel sum inside uint16.
    """

    def __init__(self, colors, threshold):
        self.thr_sq = int(float(threshold) ** 2)
        cap = self.thr_sq + 1
        if 3 * cap > np.iinfo(np.uint16).max:
            raise ValueError("Threshold too large for uint16 sums")
        v = np.arange(256, dtype=np.int64)
        self.luts = [np.stack([np.minimum((v - int(t)) ** 2, cap) for t in color], axis=-1)
                     .astype(np.uint16).reshape(256, 1, 3) for color in colors]

    def __call__(self, rgb):
        masks = []
        for lut in self.luts:
            r, g, b = cv2.split(cv2.LUT(rgb, lut))
            masks.append(cv2.compare(cv2.add(cv2.add(r, g), b), self.thr_sq, cv2.CMP_LE))
        return masks


class BoxPixelMasks:
    """The same masks in two stages: an inRange box test, then the exact distance on its pixels.

    Every pixel within the distance is inside the per-channel box, so the result is exact; the
    cost of the second stage scales with how many pixels fall inside the box.
    """

    def __init__(self, colors, threshold):
        self.thr_sq = int(float(threshold) ** 2)
        reach = int(np.floor(float(threshold)))
        self.colors = [np.array(c, dtype=np.int32) for c in colors]
        self.bounds = [(np.clip(c - reach, 0, 255).astype(np.uint8),
                        np.clip(c + reach, 0, 255).astype(np.uint8)) for c in self.colors]

    def __call__(self, rgb):
        masks = []
        for ref, (lo, hi) in zip(self.colors, self.bounds):
            mask = cv2.inRange(rgb, lo, hi)
            ys, xs = np.nonzero(mask)
            if len(ys):
                far = ((rgb[ys, xs].astype(np.int32) - ref) ** 2).sum(axis=1) > self.thr_sq
                mask[ys[far], xs[far]] = 0
            masks.append(mask)
        return masks


MASKS = dict(numpy=PixelMasks, cv2=CvPixelMasks, box=BoxPixelMasks)


def component_candidate(x, y, w, h, count, group, shape):
    """Same filters, padding and score as research_balloon_search.candidate, area = pixel count."""
    height, width = shape[:2]
    if count < 6 or min(w, h) < 4 or max(w / h, h / w) > 3 or w * h > .75 * width * height:
        return None
    pad = int(max(w, h) * .12)
    return dict(box=[x, y, x + w, y + h],
                crop_box=[max(0, x - pad), max(0, y - pad), min(width, x + w + pad),
                          min(height, y + h + pad)],
                area=float(count), color_group=group,
                proposal_score=float(np.sqrt(count) * (count / (w * h)) ** 2))


def pixel_search(rgb, masks, min_area, stride=1):
    """Components of the color masks as proposals; ``stride`` > 1 masks every stride-th pixel.

    Strided component stats are scaled back to 320x240 units (box and pixel count) before the
    unchanged candidate filters, so ``min_area`` keeps its full-resolution meaning.
    """
    if rgb.shape != (240, 320, 3) or rgb.dtype != np.uint8:
        raise ValueError("Pixel search requires 320x240 RGB uint8")
    sample = rgb if stride == 1 else np.ascontiguousarray(rgb[::stride, ::stride])
    pool = []
    for group, mask in enumerate(masks(sample)):
        # 8-connectivity matches the run-length joiner in the pixel demo.
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for x, y, w, h, count in stats[1:n] * [stride, stride, stride, stride, stride * stride]:
            if count >= min_area:
                c = component_candidate(int(x), int(y), int(w), int(h), int(count), group, rgb.shape)
                if c:
                    pool.append(c)
    return select(pool)


def process_frame(rgb, predictor, masks, min_area, stride=1):
    start = time.perf_counter()
    proposals = pixel_search(rgb, masks, min_area, stride)
    searched = time.perf_counter()
    found = []
    for proposal in proposals:
        a, b, c, d = proposal["crop_box"]
        pred = predictor.predict(rgb[b:d, a:c])
        found.append(dict(proposal, **{k: pred[k] for k in ("label", "scores", "score", "accepted")}))
    inferred = time.perf_counter()
    detections = suppress_confirmed_balloon_parts(suppress_duplicates(found))
    done = time.perf_counter()
    return dict(detections=detections, proposals=[p["box"] for p in proposals],
                proposal_groups=[p["color_group"] for p in proposals],
                processing_ms=(done - start) * 1000, search_ms=(searched - start) * 1000,
                inference_ms=(inferred - searched) * 1000, selection_ms=(done - inferred) * 1000,
                neural_calls=len(found))


def color(text):
    values = tuple(int(v) for v in text.split(","))
    if len(values) != 3 or not all(0 <= v <= 255 for v in values):
        raise argparse.ArgumentTypeError("Expected R,G,B in 0..255")
    return values


def run(args):
    root = Path(args.base)
    inputs = read_json(root / "inputs.json")
    images = [np.asarray(Image.open(root / f["path"]).convert("RGB")) for f in inputs["frames"]]
    cv2.setNumThreads(1)
    runs = []
    for model in args.model:
        predictor = Predictor(root / inputs["models"][model]["path"], allow_unvalidated=True)
        for threshold in args.threshold:
            masks = MASKS[args.masks]((args.red, args.blue), threshold)
            for rgb in images:  # warm-up
                process_frame(rgb, predictor, masks, args.min_area, args.stride)
            passes = [[process_frame(rgb, predictor, masks, args.min_area, args.stride) for rgb in images]
                      for _ in range(args.repeats)]
            frames = passes[0]
            keys = ("processing_ms", "search_ms", "inference_ms", "selection_ms")
            timing = {k: float(np.mean([r[k] for p in passes for r in p])) for k in keys}
            timing["p95_processing_ms"] = float(np.percentile(
                [r["processing_ms"] for p in passes for r in p], 95))
            runs.append(dict(model=model, threshold=threshold, min_area=args.min_area, masks=args.masks,
                             stride=args.stride,
                             timing=timing, mean_neural_calls=float(np.mean(
                                 [r["neural_calls"] for r in frames])),
                             frames=[{k: r[k] for k in ("detections", "proposals", "proposal_groups")}
                                     for r in frames]))
            print(f"{model} thr {threshold:g} stride {args.stride}: {timing['processing_ms']:.1f} ms "
                  f"(search {timing['search_ms']:.1f}, infer {timing['inference_ms']:.1f}, "
                  f"{runs[-1]['mean_neural_calls']:.2f} calls)", flush=True)
    return dict(purpose="Pixel-mask proposals + neural classifier; development frames, NOT qualification",
                deployment_approved=False, flight_commands=None, test_evaluated=False,
                colors=dict(red_balloon=list(args.red), blue_balloon=list(args.blue)),
                inputs_sha256=sha256(root / "inputs.json"), machine=platform.machine(),
                opencv=cv2.__version__, script_sha256=sha256(__file__), repeats=args.repeats,
                runs=runs)


def coverage(truth, proposals, groups):
    from dtr.tracking import iou

    return sum(any(GROUPS[g] == t["label"] and iou(t["box"], p) >= .5
                   for p, g in zip(proposals, groups)) for t in truth)


def score(base, report):
    from scripts.review_pixel_balloon_bench import panel_counts

    inputs = read_json(Path(base) / "inputs.json")
    if report["inputs_sha256"] != sha256(Path(base) / "inputs.json"):
        raise ValueError("Report used different inputs")
    frames = inputs["frames"]
    rows = []
    for run in report["runs"]:
        counts = panel_counts(frames, [f["detections"] for f in run["frames"]])
        covered = sum(coverage(f["truth"], r["proposals"], r["proposal_groups"])
                      for f, r in zip(frames, run["frames"]))
        rows.append(dict(model=run["model"], threshold=run["threshold"], timing=run["timing"],
                         mean_neural_calls=run["mean_neural_calls"], counts=counts,
                         proposal_coverage=covered, targets=sum(len(f["truth"]) for f in frames)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--model", action="append")
    parser.add_argument("--red", type=color)
    parser.add_argument("--blue", type=color)
    parser.add_argument("--threshold", type=float, action="append")
    parser.add_argument("--min-area", type=int, default=53)
    parser.add_argument("--masks", choices=sorted(MASKS), default="numpy")
    parser.add_argument("--stride", type=int, choices=(1, 2), default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", required=True)
    parser.add_argument("--score", action="store_true", help="Score an existing --output report")
    parser.add_argument("--review", help="With --score, also write the scored rows here")
    args = parser.parse_args()
    if args.score:
        rows = score(args.base, read_json(args.output))
        for row in rows:
            flat = " ".join(f"{p}:{label.split('_')[0]} {c['tp']}/{c['fp']}/{c['fn']}"
                            for p, by in row["counts"].items() for label, c in by.items())
            print(f"{row['model']} thr {row['threshold']:g}: cover {row['proposal_coverage']}/"
                  f"{row['targets']} {row['timing']['processing_ms']:.0f} ms  {flat}")
        if args.review:
            write_json(args.review, dict(
                purpose="Scored pixel-proposal runs; development frames, NOT qualification",
                deployment_approved=False, test_evaluated=False,
                report_sha256=sha256(args.output), rows=rows))
        return
    if not (args.model and args.red and args.blue and args.threshold):
        parser.error("--model, --red, --blue and --threshold are required unless --score")
    if not 1 <= args.repeats <= 10:
        parser.error("--repeats must be 1..10")
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    write_json(out, run(args))


if __name__ == "__main__":
    main()
