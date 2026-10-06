"""Bounded observational comparison. Generated replay is NOT competition footage.

Import the supplied pixel module, never its camera/server main. No actuation.
"""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import time

import cv2
import numpy as np
from PIL import Image

from dtr.data import sha256
from dtr.runtime import Predictor
from dtr.temporal import TemporalVision
from dtr.vision import observe


def load_pixel(path):
    spec = importlib.util.spec_from_file_location("pixel_candidate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scene(rgb, truth, index, count=30):
    """Same deterministic approach/translation/exposure, plus total loss/return."""
    if count // 3 <= index < count // 3 + 3:
        return np.zeros_like(rgb), [], "missing"
    h, w = rgb.shape[:2]
    phase = index / max(1, count - 1)
    scale = 1 + .15 * phase
    dx, dy = 8 * math.sin(phase * math.pi), 3 * phase
    transform = np.float32([[scale, 0, (1-scale)*w/2+dx],
                            [0, scale, (1-scale)*h/2+dy]])
    frame = cv2.warpAffine(rgb, transform, (w, h))
    frame = np.clip(frame.astype(np.float32) * (1-.15*phase), 0, 255).astype(np.uint8)
    labels = []
    for row in truth:
        a, b, c, d = row["box"]
        x, y = transform[0, 2], transform[1, 2]
        box = [max(0, a*scale+x), max(0, b*scale+y),
               min(w, c*scale+x), min(h, d*scale+y)]
        if box[2] > box[0] and box[3] > box[1]:
            labels.append(dict(label=row["label"], box=[float(v) for v in box]))
    return frame, labels, "present"


def next_index(elapsed, fps, previous):
    return max(previous+1, math.floor(elapsed*fps))


def summarize(rows, matching):
    totals = dict(tp=0, fp=0, fn=0)
    target_frames = hit_frames = 0
    for row in rows:
        truth = [r["box"] for r in row["truth"] if r["label"].startswith("yellow_")]
        counts = matching(row["boxes"], truth)
        for key in totals:
            totals[key] += counts[key]
        target_frames += bool(truth)
        hit_frames += bool(counts["tp"])
    tp, fp, fn = (totals[k] for k in ("tp", "fp", "fn"))
    times = [r["processing_ms"] for r in rows]
    ages = [r["result_age_ms"] for r in rows if "result_age_ms" in r]
    return dict(frames=len(rows), **totals, precision=tp/(tp+fp) if tp+fp else 0,
                recall=tp/(tp+fn) if tp+fn else 0,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,
                target_present_frames=target_frames, frames_with_correct_target=hit_frames,
                any_target_success=hit_frames/target_frames if target_frames else None,
                mean_ms=float(np.mean(times)), p95_ms=float(np.percentile(times, 95)),
                processing_fps=1000/float(np.mean(times)),
                p95_result_age_ms=float(np.percentile(ages, 95)) if ages else None,
                skipped_frames=sum(r.get("skipped", 0) for r in rows),
                over_100ms=sum(t > 100 for t in times),
                missing_frame_detections=sum(len(r["boxes"]) for r in rows if r.get("phase") == "missing"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", required=True)
    p.add_argument("--pixel", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--mode", choices=["static", "paced"], default="static")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--fps", type=float, default=10)
    p.add_argument("--repeat", type=int, default=2)
    args = p.parse_args()
    if args.limit < 0 or not 1 <= args.repeat <= 4 or not 1 <= args.fps <= 30:
        p.error("invalid bounded run settings")
    output, base = Path(args.output), Path(args.base)
    if output.exists() or output.with_suffix(".jsonl").exists():
        raise FileExistsError(output)
    # Frozen helper imported from the existing comparison bundle.
    import sys
    sys.path.insert(0, str(base))
    from pi_compare import localization_counts
    from pi_temporal_bench import select_clips
    manifest = json.loads((base/"frames.json").read_text())
    if manifest["split"] != "development_validation":
        raise ValueError("Only development inputs permitted")
    for row in manifest["frames"]:
        if sha256(base/row["path"]) != row["sha256"]:
            raise ValueError("Input checksum mismatch")
    cv2.setNumThreads(1)
    pixel = load_pixel(args.pixel)
    predictor = Predictor(args.model, allow_unvalidated=True)
    tracker = TemporalVision(predictor, budget=4, difference_backend="native")
    memory = pixel.GoalMemory()
    records = {"pixel": [], "context": []}

    def infer(name, rgb, temporal):
        start = time.perf_counter()
        if name == "pixel":
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            goal, _ = memory.update(bgr)
            boxes = [] if goal is None else [[int(goal["box"][0]), int(goal["box"][1]),
                                              int(goal["box"][2])+1, int(goal["box"][3])+1]]
            obs = []
        else:
            result = tracker.observe(rgb) if temporal else observe(
                rgb, predictor, limit=12, profile="balloon_components", duplicate_policy="nested")
            obs = result["observations"]
            boxes = [o["box"] for o in obs if o["accepted"] and o["label"].startswith("yellow_")]
        return dict(boxes=boxes, observations=obs, processing_ms=(time.perf_counter()-start)*1000)

    sources = manifest["frames"]
    if args.mode == "paced":
        sources = select_clips(sources, [c for c in predictor.metadata["classes"] if c != "background"])
    sources = sources[:args.limit or None]
    # Warmups excluded; reset all state afterward.
    first = np.array(Image.open(base/sources[0]["path"]).convert("RGB"))
    for name in records:
        infer(name, first, False)
    elapsed_by_method = {name: 0. for name in records}
    with output.with_suffix(".jsonl").open("x") as log:
        for clip, source in enumerate(sources):
            rgb = np.array(Image.open(base/source["path"]).convert("RGB"))
            if args.mode == "static":
                memory = pixel.GoalMemory()
                for name in list(records)[::1 if clip % 2 == 0 else -1]:
                    row = infer(name, rgb, False)
                    row.update(source=source["path"], truth=source["truth"], method=name)
                    records[name].append(row)
                    log.write(json.dumps(row)+"\n")
            else:
                # Pre-generation excluded; each method sees the same 3-second timeline.
                count = int(3*args.fps)
                frames = [scene(rgb, source["truth"], i, count) for i in range(count)]
                for repeat in range(args.repeat):
                    for name in list(records)[::1 if (clip+repeat) % 2 == 0 else -1]:
                        memory, tracker = pixel.GoalMemory(), TemporalVision(
                            predictor, budget=4, difference_backend="native")
                        start, previous = time.monotonic(), -1
                        while True:
                            index = next_index(time.monotonic()-start, args.fps, previous)
                            if index >= count:
                                break
                            due = start+index/args.fps
                            if due > time.monotonic():
                                time.sleep(due-time.monotonic())
                            frame, truth, phase = frames[index]
                            row = infer(name, frame, True)
                            row.update(source=source["path"], method=name, clip=clip, repeat=repeat,
                                       frame=index, truth=truth, phase=phase, skipped=index-previous-1,
                                       result_age_ms=(time.monotonic()-due)*1000)
                            records[name].append(row)
                            log.write(json.dumps(row)+"\n")
                            previous = index
                        elapsed_by_method[name] += time.monotonic()-start
            if clip % 25 == 0 or args.mode == "paced":
                print("Completed", clip+1, "/", len(sources), flush=True)
    result = dict(mode=args.mode, host=platform.node(), machine=platform.machine(),
                  source_frames=len(sources), image_size=manifest["size"],
                  pixel_sha256=sha256(args.pixel), model_sha256=sha256(args.model),
                  library=os.environ.get("DTR_TFLITE_LIBRARY"),
                  pixel_backend=pixel.backend_name(),
                  pixel_settings={k:getattr(pixel, k) for k in (
                      "TARGET_COLOR", "COLOR_THRESHOLD", "MIN_SHAPE_SIZE", "PICK", "USE_MEMORY",
                      "ADAPTIVE_STRIDE", "DOWNSAMPLE_FACTOR", "ROI_EXPAND")},
                  flight_commands=None, independent_test=False,
                  scope="Yellow localization; supplied pixel method selects one goal, context can emit multiple. "
                        "Static resets pixel memory per image. Paced uses actual elapsed time and drops frames; "
                        "generated translation/zoom/exposure and blackouts are not competition video. "
                        "Accuracy is on each method's processed frames, not a common temporal sample in paced mode.",
                  results={k:summarize(v, localization_counts) for k,v in records.items()})
    if args.mode == "paced":
        result.update(source_fps=args.fps, repeats=args.repeat, elapsed_by_method=elapsed_by_method)
        for name in records:
            result["results"][name]["delivered_fps"] = len(records[name])/elapsed_by_method[name]
            available = len(sources)*int(3*args.fps)*args.repeat
            result["results"][name]["timeline_frames"] = available
            result["results"][name]["skipped_frames"] = available-len(records[name])
    with output.open("x") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
