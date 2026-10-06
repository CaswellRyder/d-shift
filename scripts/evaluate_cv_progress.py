"""Paired, hash-checked CV development ablation; never promotes models or uses test data."""
import argparse
import json
from pathlib import Path
import platform
import shutil
import sys

import cv2
import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, write_json
from dtr.runtime import Predictor
from dtr.tracking import iou
from dtr.vision import observe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="runs/pi-comparison-20261005")
    parser.add_argument("--model", action="append", required=True, help="name=model.tflite")
    parser.add_argument("--profile", action="append", choices=("balloon_components", "goal_edges"),
                        help="Repeat for paired profiles; default compares baseline and edge search")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    base, out = Path(args.base).resolve(), Path(args.output).resolve()
    manifest = read_json(base / "frames.json")
    if manifest.get("split") != "development_validation" or manifest.get("size") != [320, 240]:
        raise ValueError("Require the frozen 320x240 development frames")
    models = {}
    for spec in args.model:
        name, path = spec.split("=", 1)
        if (not name or not all(c.isalnum() or c in "-_" for c in name) or name in models):
            raise ValueError("Unique alphanumeric model name required")
        models[name] = Predictor(path, allow_unvalidated=True)
        if models[name].metadata["task"] != "goal":
            raise ValueError("Goal models only")
    # Verify all selected files before writing evidence. No reserved-test directory is read.
    for row in manifest["frames"]:
        source = (base / row["path"]).resolve()
        if not source.is_relative_to(base) or sha256(source) != row["sha256"]:
            raise ValueError("Frozen input hash/path mismatch")
    out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    sources = ["src/dtr/vision.py", "src/dtr/goal_search.py", "src/dtr/models.py",
               "src/dtr/runtime.py", "src/dtr/goal_evidence.py", "src/dtr/tracking.py",
               "scripts/distill_pi_student.py", "scripts/evaluate_cv_progress.py"]
    for name in sources:
        target = out / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, target)
    sys.path.insert(0, str(base))
    from pi_compare import localization_counts, metrics
    cv2.setNumThreads(1)
    reports = {}
    profiles = list(dict.fromkeys(args.profile or ("balloon_components", "goal_edges")))
    variants = [(name, profile) for name in models for profile in profiles]
    stats = {}
    streams = {}
    for name, profile in variants:
        key = name + "-" + profile
        classes = {label: dict(tp=0, fp=0, fn=0) for label in models[name].metadata["classes"]
                   if label != "background"}
        stats[key] = dict(classes=classes, covered={label: 0 for label in classes},
                          totals={label: 0 for label in classes}, times=[], stages=[],
                          candidates=0, edge_candidates=0, tiny_total=0, tiny_covered=0)
        streams[key] = (out / (key + ".frames.jsonl")).open("x")
        for row in manifest["frames"][:3]:
            with Image.open(base / row["path"]) as image:
                observe(np.array(image.convert("RGB")), models[name], limit=12, profile=profile)
    try:
        for index, frame in enumerate(manifest["frames"]):
            with Image.open(base / frame["path"]) as image:
                rgb = np.array(image.convert("RGB"))
            # Alternate variant order, avoiding an always-first timing advantage.
            for name, profile in (variants if index % 2 == 0 else variants[::-1]):
                key = name + "-" + profile
                result = observe(rgb, models[name], limit=12, profile=profile)
                state = stats[key]
                rows = result["observations"]
                accepted = [r for r in rows if r["accepted"]]
                state["times"].append(result["processing_ms"])
                state["stages"].append(result["stage_ms"])
                state["candidates"] += len(rows)
                state["edge_candidates"] += sum(r.get("proposal_source") == "grayscale_edges" for r in rows)
                for label, cell in state["classes"].items():
                    counts = localization_counts([r["box"] for r in accepted if r["label"] == label],
                                                 [r["box"] for r in frame["truth"] if r["label"] == label])
                    for field in cell:
                        cell[field] += counts[field]
                for target in frame["truth"]:
                    label, box = target["label"], target["box"]
                    covered = any(iou(box, row["box"]) >= .5 for row in rows)
                    state["totals"][label] += 1
                    state["covered"][label] += covered
                    if max(box[2]-box[0], box[3]-box[1]) < 8:
                        state["tiny_total"] += 1
                        state["tiny_covered"] += covered
                streams[key].write(json.dumps(dict(frame=frame["path"], **result)) + "\n")
            if index % 100 == 0:
                print("Frames", index+1, "of", len(manifest["frames"]), flush=True)
    finally:
        for stream in streams.values():
            stream.close()
    for name, profile in variants:
        key = name + "-" + profile
        state = stats[key]
        micro = {field: sum(c[field] for c in state["classes"].values()) for field in ("tp", "fp", "fn")}
        reports[key] = dict(
            model_sha256=models[name].metadata["sha256"], profile=profile, budget=12,
            micro=metrics(micro), classes={k: metrics(v) for k, v in state["classes"].items()},
            coverage=dict(covered=sum(state["covered"].values()), total=sum(state["totals"].values()),
                          by_class=state["covered"], totals=state["totals"],
                          tiny_covered=state["tiny_covered"], tiny_total=state["tiny_total"]),
            candidates=state["candidates"], edge_candidates=state["edge_candidates"],
            mean_ms=float(np.mean(state["times"])), p95_ms=float(np.percentile(state["times"], 95)),
            mean_stage_ms={k: float(np.mean([s[k] for s in state["stages"]])) for k in state["stages"][0]})
    report = dict(results=reports, frames=len(manifest["frames"]), host=platform.node(),
                  machine=platform.machine(), frame_manifest_sha256=sha256(base / "frames.json"),
                  source_sha256={name: sha256(out / "source" / name) for name in sources},
                  scope="Repeated development scenes; no independent field accuracy or Pi speed claim",
                  timing_scope="Search through suppression; excludes capture/load/render/transport",
                  deployment_approved=False, test_evaluated=False, flight_commands=None)
    write_json(out / "report.json", report)
    print(json.dumps(reports, indent=2), flush=True)


if __name__ == "__main__":
    main()
