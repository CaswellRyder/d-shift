"""Isolated runtime and frame experiments; never alters the original bundle."""
import argparse
import json
from pathlib import Path
import platform

import cv2
import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, write_json
from dtr.runtime import Predictor
from dtr.vision import PROPOSAL_PROFILES, observe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["crops", "frames"], default="crops")
    parser.add_argument("--model", default="goal.float.tflite")
    parser.add_argument("--budget", type=int, default=12)
    parser.add_argument("--profile", choices=PROPOSAL_PROFILES, default="balloon_components")
    parser.add_argument("--golden",default="float-golden.json")
    args = parser.parse_args()
    base, output = Path(args.base), Path(args.output)
    if output.exists() or output.with_suffix(".frames.jsonl").exists():
        raise FileExistsError(output)
    cv2.setNumThreads(1)
    report = dict(host=platform.node(), machine=platform.machine(), mode=args.mode, flight_commands=None,
                  deployment_approved=False, test_evaluated=False)
    if args.mode == "crops":
        golden = read_json(args.golden)
        results = {}
        for name, path in [("int8", base / "models/goal.tflite"), ("float32", Path(args.model))]:
            predictor = Predictor(path, allow_unvalidated=True)
            images = [np.array(Image.open(base / row["path"]).convert("RGB")) for row in golden]
            for rgb in images[:3]:
                predictor.predict(rgb)
            durations, deltas, labels = [], [], []
            for rgb, row in zip(images, golden):
                got = predictor.predict(rgb)
                durations.append(got["latency_ms"])
                if name == "float32":
                    deltas.append(float(np.max(np.abs(np.array(got["scores"])-row["scores"])) ))
                    labels.append(got["label"] == row["label"])
            if name == "float32" and (not all(labels) or max(deltas) > .001):
                raise ValueError("FP32 reference parity failed")
            results[name] = dict(mean_ms=float(np.mean(durations)), p95_ms=float(np.percentile(durations,95)),
                                 model_sha256=sha256(path), max_score_delta=max(deltas,default=None))
            print(name, results[name], flush=True)
        report["results"] = results
    else:
        # The original scoring helper is loaded, not modified.
        import sys
        sys.path.append(str(base.resolve()))
        from pi_compare import localization_counts, metrics
        predictor = Predictor(args.model, allow_unvalidated=True)
        counts = dict(tp=0,fp=0,fn=0)
        by_class = {c: dict(tp=0,fp=0,fn=0) for c in predictor.metadata["classes"] if c != "background"}
        times = []
        rows = read_json(base / "frames.json")["frames"]
        for row in rows[:5]:
            rgb = np.array(Image.open(base / row["path"]).convert("RGB"))
            observe(rgb,predictor,limit=args.budget,profile=args.profile,duplicate_policy="nested")
        with output.with_suffix(".frames.jsonl").open("x") as stream:
            for index, row in enumerate(rows):
                rgb = np.array(Image.open(base / row["path"]).convert("RGB"))
                result = observe(rgb,predictor,limit=args.budget,profile=args.profile,duplicate_policy="nested")
                accepted = [r for r in result["observations"] if r["accepted"]]
                times.append(result["processing_ms"])
                matched = localization_counts([r["box"] for r in accepted if r["label"].startswith("yellow_")],
                                               [r["box"] for r in row["truth"] if r["label"].startswith("yellow_")])
                for key in counts:
                    counts[key] += matched[key]
                for label in by_class:
                    cell = localization_counts([r["box"] for r in accepted if r["label"] == label],
                                                [r["box"] for r in row["truth"] if r["label"] == label])
                    for key in cell:
                        by_class[label][key] += cell[key]
                stream.write(json.dumps(dict(frame=row["path"], **result))+"\n")
                if index % 100 == 0:
                    print("Frames",index+1,"of",len(rows),flush=True)
        report.update(frames=len(rows), split="development_validation", budget=args.budget,
                      proposal_profile=args.profile,
                      model_sha256=sha256(args.model), yellow_localization=metrics(counts),
                      classes={k:metrics(v) for k,v in by_class.items()},
                      mean_ms=float(np.mean(times)),p50_ms=float(np.median(times)),
                      p95_ms=float(np.percentile(times,95)),processing_fps=1000/float(np.mean(times)),
                      timing_scope="detection only; excludes camera, load, render; not live FPS")
    write_json(output,report)
    print(report,flush=True)


if __name__ == "__main__":
    main()
