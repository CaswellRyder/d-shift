"""Paired development-frame comparison: fixed proposals, low vs high detail.

No oracle boxes, no threshold tuning, no reserved-test images. Predecoded images;
timings exclude source decode/camera/transport. Not an ARMv6 performance benchmark.
"""
import argparse
import json
from pathlib import Path
import platform

import cv2
import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, write_json
from dtr.detail_vision import observe_detail
from dtr.runtime import Predictor
from pi_compare import localization_counts, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="runs/pi-comparison-20261005")
    parser.add_argument("--source", default="data/roboflow-dtr-v10-grouped/coco/valid")
    parser.add_argument("--model", action="append", required=True, help="name=model.tflite")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() or output.with_suffix(".frames.jsonl").exists():
        raise FileExistsError(output)
    base, source = Path(args.base), Path(args.source).resolve()
    frames = read_json(base / "frames.json")
    if (frames["split"] != "development_validation" or frames["size"] != [320,240]
            or sha256(source / "_annotations.coco.json") != frames["annotation_sha256"]):
        raise ValueError("Wrong development frames/annotations")
    predictors, identities = {}, {}
    for item in args.model:
        name, path = item.split("=", 1)
        if name in predictors:
            raise ValueError("Duplicate model name")
        predictors[name] = Predictor(path, True)
        identities[name] = dict(path=path, sha256=sha256(path))
    classes = next(iter(predictors.values())).metadata["classes"]
    if any(p.metadata["task"] != "goal" or p.metadata["classes"] != classes for p in predictors.values()):
        raise ValueError("Taxonomy mismatch")
    cv2.setNumThreads(1)
    totals, durations = {}, {}
    for model in predictors:
        for detail in ("320x240", "640x480"):
            name = f"{model}/{detail}"
            totals[name] = {c: dict(tp=0,fp=0,fn=0) for c in classes if c != "background"}
            totals[name]["yellow_localization"] = dict(tp=0,fp=0,fn=0)
            durations[name] = []
    output.parent.mkdir(parents=True, exist_ok=True)
    source_hashes = {}
    with output.with_suffix(".frames.jsonl").open("x") as stream:
        for index, frame in enumerate(frames["frames"]):
            src = (source / frame["source"]).resolve()
            low_path = (base / frame["path"]).resolve()
            if not src.is_relative_to(source) or not low_path.is_relative_to(base.resolve()):
                raise ValueError("Frame escapes dataset root")
            if sha256(low_path) != frame["sha256"]:
                raise ValueError("Frozen frame changed")
            source_hashes[frame["source"]] = sha256(src)
            with Image.open(low_path) as im:
                low = np.array(im.convert("RGB"))
            with Image.open(src) as im:
                high = cv2.resize(np.array(im.convert("RGB")), (640,480))
            methods = [(model, detail, rgb, scale) for model in predictors
                       for detail, rgb, scale in (("320x240", low, 1), ("640x480", high, 2))]
            if index % 2:
                methods.reverse()
            for model, detail, rgb, scale in methods:
                name = f"{model}/{detail}"
                result = observe_detail(rgb, predictors[model], scan_rgb=low)
                predictions = [dict(o, box=[v/scale for v in o["box"]])
                               for o in result["observations"] if o["accepted"]]
                for label, counts in totals[name].items():
                    def relevant(s):
                        return s.startswith("yellow_") if label == "yellow_localization" else s == label
                    cell = localization_counts([o["box"] for o in predictions if relevant(o["label"])],
                                               [t["box"] for t in frame["truth"] if relevant(t["label"])])
                    for key, value in cell.items():
                        counts[key] += value
                durations[name].append(result["processing_ms"])
                stream.write(json.dumps(dict(method=name, frame=frame["path"], **result)) + "\n")
            if index % 100 == 0:
                print(index+1, "of", len(frames["frames"]), flush=True)
    results = {}
    for name, counts in totals.items():
        same_class = {k: sum(c[k] for label,c in counts.items() if label != "yellow_localization")
                      for k in ("tp", "fp", "fn")}
        results[name] = dict(classes={c: metrics(v) for c,v in counts.items()},
                             same_class_micro=metrics(same_class),
                             mean_ms=float(np.mean(durations[name])),
                             p95_ms=float(np.percentile(durations[name], 95)))
    write_json(output, dict(results=results, models=identities, host=platform.node(),
                           machine=platform.machine(), frames=len(frames["frames"]),
                           frames_sha256=sha256(base / "frames.json"), source_image_sha256=source_hashes,
                           script_sha256=sha256(__file__),
                           detail_vision_sha256=sha256(Path(__file__).resolve().parents[1] / "src/dtr/detail_vision.py"),
                           proposal_budget=12, scan_size=[320,240], test_evaluated=False,
                           deployment_approved=False, timing_scope="Mac predecoded detection only, includes cold start; not live FPS",
                           scope="Same frozen development proposals; only crop source detail changes; correlated reused data"))
    print({name: value["same_class_micro"] for name,value in results.items()}, flush=True)


if __name__ == "__main__":
    main()
