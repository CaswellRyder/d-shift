"""Research-only minimum visual context for tiny proposals; no box/threshold change."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, write_json
from dtr.runtime import Predictor
from dtr.vision import suppress_duplicates
from scripts.evaluate_red_blue_development import LABELS, detection_counts, metrics
from scripts.pi_balloon_search_bench import check_predictions
from scripts.pi_red_blue_bench import verify_bundle
from scripts.research_balloon_search import experimental, suppress_confirmed_balloon_parts
from scripts.research_predictor import classify_candidates


def minimum_context(candidates, shape):
    """For crops smaller than 16x16, add surroundings, never extrapolate pixels.

    Fixed first experiment, not a threshold sweep. Tiny candidate boxes remain
    available: this is not a minimum-object-size rejection filter.
    """
    height, width = shape[:2]
    if min(height, width) < 16:
        raise ValueError("Context experiment requires an image at least 16x16")
    result = []
    for candidate in candidates:
        row = dict(candidate)
        a, b, c, d = row["crop_box"]
        if (
            any(type(v) is not int for v in (a, b, c, d))
            or not 0 <= a < c <= width
            or not 0 <= b < d <= height
        ):
            raise ValueError("Invalid candidate crop")
        if max(c - a, d - b) < 16:
            x = min(max((a + c - 16) // 2, 0), width - 16)
            y = min(max((b + d - 16) // 2, 0), height - 16)
            row["crop_box"] = [x, y, x + 16, y + 16]
        result.append(row)
    return result


def classify(rgb, predictor, candidates):
    return suppress_confirmed_balloon_parts(
        suppress_duplicates(classify_candidates(predictor, rgb, candidates))
    )


def evaluate(base, sample_dirs=()):
    base = Path(base)
    verify_bundle(base)
    inputs, golden = read_json(base / "inputs.json"), read_json(base / "search-golden.json")
    frames = []
    for frame in inputs["frames"]:
        frames.append((frame, np.asarray(Image.open(base / frame["path"]).convert("RGB"))))
    # Live images remain diagnostics only, never fabricated as labeled scenes.
    live = []
    for directory in sample_dirs:
        root = Path(directory).resolve()
        report = read_json(root / "report.json")
        for name, digest in report["files"].items():
            if not name.endswith("-raw.png"):
                continue
            path = (root / name).resolve()
            if not path.is_relative_to(root) or sha256(path) != digest:
                raise ValueError("Live image identity changed")
            rgb = np.asarray(Image.open(path).convert("RGB"))
            live.append((dict(source=f"{root.name}/{name}", sha256=digest), rgb))
    results = {}
    for name, model in inputs["models"].items():
        predictor = Predictor(base / model["path"], allow_unvalidated=True)
        panels = {}
        for variant in ("original", "minimum_context16"):
            counts = {}
            outputs = []
            for i, (frame, rgb) in enumerate(frames):
                candidates = experimental(rgb, "mser")
                if variant != "original":
                    candidates = minimum_context(candidates, rgb.shape)
                detections = classify(rgb, predictor, candidates)
                if variant == "original":
                    check_predictions(detections, golden[name]["mser"][i]["detections"])
                found = detection_counts(frame["truth"], detections)
                panel = counts.setdefault(
                    frame["panel"], {k: dict(tp=0, fp=0, fn=0) for k in LABELS}
                )
                for label in LABELS:
                    for key in ("tp", "fp", "fn"):
                        panel[label][key] += found[label][key]
                outputs.append(
                    dict(
                        source=frame["source"],
                        panel=frame["panel"],
                        counts=found,
                        detections=detections,
                    )
                )
            diagnostics = []
            for record, rgb in live:
                candidates = experimental(rgb, "mser")
                if variant != "original":
                    candidates = minimum_context(candidates, rgb.shape)
                diagnostics.append(dict(**record, detections=classify(rgb, predictor, candidates)))
            panels[variant] = dict(
                metrics={k: metrics(v) for k, v in counts.items()},
                frames=outputs,
                live_samples=diagnostics,
            )
        results[name] = dict(model_sha256=sha256(base / model["path"]), variants=panels)
    return dict(
        scope="Fixed minimum-context development ablation; live samples are unlabeled diagnostics",
        results=results,
        script_sha256=sha256(__file__),
        bundle_sha256=sha256(base / "bundle.json"),
        test_evaluated=False,
        deployment_approved=False,
        training_performed=False,
        pi_timing_measured=False,
        live_sample_accuracy_measured=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--live-samples", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    report = evaluate(args.base, args.live_samples)
    write_json(args.output, report)
    print(
        {
            k: {v: r["metrics"] for v, r in d["variants"].items()}
            for k, d in report["results"].items()
        }
    )
