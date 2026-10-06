"""Compare frozen methods on identical RGB frames. No camera or actuation.

Shared accuracy task is YELLOW GOAL LOCALIZATION, ignoring goal shape. The
baseline cannot identify shape or other colors with its current configuration.
"""
import argparse
import importlib.util
import platform
from pathlib import Path
import resource
import time

import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, write_json
from dtr.runtime import Predictor
from dtr.tracking import iou
from dtr.vision import observe


def load_baseline(path):
    spec = importlib.util.spec_from_file_location("comparison_baseline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def localization_counts(predictions, truth):
    """Maximum-cardinality IoU matching; baseline has no confidence ordering."""
    edges = [[j for j, box in enumerate(truth) if iou(pred, box) >= 0.5] for pred in predictions]
    assigned = {}

    def match(index, seen):
        for target in edges[index]:
            if target in seen:
                continue
            seen.add(target)
            if target not in assigned or match(assigned[target], seen):
                assigned[target] = index
                return True
        return False

    tp = sum(match(index, set()) for index in range(len(predictions)))
    return dict(tp=tp, fp=len(predictions)-tp, fn=len(truth)-tp)


def metrics(counts):
    tp, fp, fn = (counts[k] for k in ("tp", "fp", "fn"))
    return {**counts, "precision": tp/(tp+fp) if tp+fp else 0.,
            "recall": tp/(tp+fn) if tp+fn else 0.,
            "f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.}


def verify_bundle(root):
    receipt = read_json(root / "bundle.json")
    for name, digest in receipt["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or sha256(path) != digest:
            raise ValueError(f"Bundle checksum mismatch: {name}")
    return receipt


def golden_check(root):
    results = []
    for task in ("goal", "balloon"):
        predictor = Predictor(root / "models" / f"{task}.tflite", allow_unvalidated=True)
        for row in read_json(root / "golden.json"):
            if row["task"] != task:
                continue
            got = predictor.predict(root / row["path"])
            delta = float(np.max(np.abs(np.array(got["scores"])-row["scores"])))
            if got["label"] != row["label"] or delta > .03:
                raise ValueError(f"Golden crop mismatch {row['path']}: {delta}")
            results.append(dict(path=row["path"], max_score_delta=delta))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", default=".")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0, help="0 means all frames")
    parser.add_argument("--golden-only", action="store_true")
    args = parser.parse_args()
    root, output = Path(args.bundle), Path(args.output)
    if output.exists() or output.with_suffix(".frames.jsonl").exists():
        raise FileExistsError(output)
    if args.limit < 0:
        parser.error("limit must be nonnegative")
    verify_bundle(root)
    golden = golden_check(root)
    header = dict(host=platform.node(), machine=platform.machine(), python=platform.python_version(),
                  bundle_sha256=sha256(root / "bundle.json"), golden_crops=golden,
                  flight_commands=None, deployment_approved=False)
    if args.golden_only:
        write_json(output, header)
        print("Golden crops passed:", len(golden), flush=True)
        return
    baseline = load_baseline(root / "baseline.py")
    detector = baseline.Detector(baseline.TARGET_COLOR_RGB, baseline.COLOR_THRESHOLD,
                                 baseline.MIN_SHAPE_SIZE, backend="numpy")
    predictor = Predictor(root / "models/goal.tflite", allow_unvalidated=True)

    def baseline_detect(rgb):
        boxes, _ = detector.detect(rgb)
        # Baseline coordinates have inclusive right/bottom; common IoU is half-open.
        return [[int(left), int(top), int(right)+1, int(bottom)+1]
                for left, top, right, bottom in boxes]

    def alternative_detect(rgb):
        observations = observe(rgb, predictor, limit=12, profile="balloon_components",
                               duplicate_policy="nested")["observations"]
        return [o["box"] for o in observations if o["accepted"] and o["label"].startswith("yellow_")]

    functions = {"baseline_rgb_components": baseline_detect, "distilled_int8": alternative_detect}
    corpus = read_json(root / "frames.json")
    frames = corpus["frames"][:args.limit or None]
    for frame in frames[:5]:
        with Image.open(root / frame["path"]) as opened:
            rgb = np.array(opened.convert("RGB"))
        for function in functions.values():
            function(rgb)
    totals = {name: dict(tp=0, fp=0, fn=0) for name in functions}
    durations = {name: [] for name in functions}
    output.parent.mkdir(parents=True, exist_ok=True)
    import json
    with output.with_suffix(".frames.jsonl").open("x") as log:
        for index, frame in enumerate(frames):
            with Image.open(root / frame["path"]) as opened:
                rgb = np.array(opened.convert("RGB"))
            truth = [r["box"] for r in frame["truth"] if r["label"].startswith("yellow_")]
            names = list(functions)
            if index % 2:
                names.reverse()
            for name in names:
                start = time.perf_counter()
                predictions = functions[name](rgb)
                duration = (time.perf_counter()-start)*1000
                counts = localization_counts(predictions, truth)
                for key, value in counts.items():
                    totals[name][key] += value
                durations[name].append(duration)
                log.write(json.dumps(dict(frame=frame["path"], method=name, boxes=predictions,
                                          processing_ms=duration, **counts))+"\n")
            if index % 50 == 0:
                print("Compared", index+1, "of", len(frames), flush=True)
    results = {}
    for name, values in durations.items():
        results[name] = dict(**metrics(totals[name]), mean_ms=float(np.mean(values)),
                            p50_ms=float(np.median(values)), p95_ms=float(np.percentile(values, 95)),
                            processing_fps=1000/float(np.mean(values)))
    write_json(output, {**header, "frames": len(frames), "split": corpus["split"],
        "task": "yellow goal localization, shape ignored, IoU >= 0.5, maximum-cardinality matching",
        "image_size": corpus["size"], "warmup_frames": min(5, len(frames)), "results": results,
        "baseline": dict(color=baseline.TARGET_COLOR_RGB, threshold=baseline.COLOR_THRESHOLD,
                         min_size=baseline.MIN_SHAPE_SIZE, backend="numpy",
                         original_size=baseline.WORKING_SIZE, shape_classification="not supported"),
        "alternative": dict(proposals=12, threshold=predictor.metadata["threshold"],
                            runtime=type(predictor.interpreter).__module__,
                            runtime_version=getattr(predictor.interpreter, "version", None)),
        "timing_scope": "Same predecoded RGB frames; detection only, 1 OpenCV thread; excludes capture, load, rendering, transport, imports and model initialization; not live FPS",
        "rss_scope": "Shared comparison process high-water RSS, not per-method memory",
        "process_maxrss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "caveat": corpus["caveat"], "test_evaluated": False})
    print(results, flush=True)


if __name__ == "__main__":
    import cv2
    cv2.setNumThreads(1)
    main()
