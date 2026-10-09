"""Fixed bright-only MSER ablation, scored as a changed policy, never exactness."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, write_json
from dtr.native_balloon_regions import NativeBalloonRegions
from dtr.runtime import Predictor
from dtr.tracking import iou
from dtr.vision import suppress_duplicates
from scripts.evaluate_red_blue_development import LABELS, detection_counts, metrics
from scripts.pi_balloon_search_bench import check_predictions
from scripts.pi_red_blue_bench import verify_bundle
from scripts.research_balloon_search import suppress_confirmed_balloon_parts
from scripts.research_balloon_search_fast import search_fast
from scripts.research_predictor import classify_candidates


def evaluate(base, library):
    base = Path(base)
    verify_bundle(base)
    inputs, golden = read_json(base / "inputs.json"), read_json(base / "search-golden.json")
    native = NativeBalloonRegions(library, direct=True)
    frames = []
    for row in inputs["frames"]:
        rgb = np.asarray(Image.open(base / row["path"]).convert("RGB"))
        frames.append((row, rgb))
    results = {}
    for name, model in inputs["models"].items():
        predictor = Predictor(base / model["path"], allow_unvalidated=True)
        variants = {}
        for variant in ("both_passes", "bright_only"):
            totals, outputs = {}, []
            for index, (row, rgb) in enumerate(frames):
                candidates = search_fast(rgb, native, bright_only=variant == "bright_only")
                detections = suppress_confirmed_balloon_parts(
                    suppress_duplicates(classify_candidates(predictor, rgb, candidates))
                )
                if variant == "both_passes":
                    check_predictions(detections, golden[name]["mser"][index]["detections"])
                counts = detection_counts(row["truth"], detections)
                panel = totals.setdefault(row["panel"], {k: dict(tp=0, fp=0, fn=0) for k in LABELS})
                for label in LABELS:
                    for key in ("tp", "fp", "fn"):
                        panel[label][key] += counts[label][key]
                coverage = [
                    dict(
                        label=t["label"],
                        box=t["box"],
                        best_iou=max((iou(t["box"], c["box"]) for c in candidates), default=0.0),
                    )
                    for t in row["truth"]
                ]
                outputs.append(
                    dict(
                        source=row["source"],
                        panel=row["panel"],
                        counts=counts,
                        candidates=len(candidates),
                        coverage=coverage,
                        detections=detections,
                    )
                )
            variants[variant] = dict(
                metrics={k: metrics(v) for k, v in totals.items()}, frames=outputs
            )
        results[name] = dict(model_sha256=sha256(base / model["path"]), variants=variants)
    return dict(
        scope="Changed proposal policy on reused development scenes, not qualification",
        results=results,
        bundle_sha256=sha256(base / "bundle.json"),
        script_sha256=sha256(__file__),
        search_sha256=sha256("scripts/research_balloon_search_fast.py"),
        native_receipt=read_json(Path(library).with_suffix(".json")),
        policy="Only brighter-to-darker pass on each opponent-color plane; unchanged other settings",
        test_evaluated=False,
        deployment_approved=False,
        training_performed=False,
        pi_timing_measured=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "library", "output"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    report = evaluate(args.base, args.library)
    write_json(args.output, report)
    print(
        {
            k: {v: r["metrics"] for v, r in d["variants"].items()}
            for k, d in report["results"].items()
        }
    )
