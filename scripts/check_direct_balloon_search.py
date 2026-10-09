"""Check direct-pointer search against unchanged proposals and golden predictions."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, write_json
from dtr.native_balloon_regions import NativeBalloonRegions
from dtr.runtime import Predictor
from scripts.pi_balloon_search_bench import check_predictions, process_frame
from scripts.pi_red_blue_bench import verify_bundle
from scripts.research_balloon_search_fast import search_fast


def check(base, library):
    base = Path(base)
    verify_bundle(base)
    inputs, golden = read_json(base / "inputs.json"), read_json(base / "search-golden.json")
    old, new = NativeBalloonRegions(library), NativeBalloonRegions(library, direct=True)
    images = [np.asarray(Image.open(base / r["path"]).convert("RGB")) for r in inputs["frames"]]
    for rgb in images:
        if search_fast(rgb, old) != search_fast(rgb, new):
            raise ValueError("Direct-pointer proposals changed")
    results = {}
    for name, model in inputs["models"].items():
        predictor = Predictor(base / model["path"], allow_unvalidated=True)
        deltas = []
        for rgb, reference in zip(images, golden[name]["mser"], strict=True):
            control = process_frame(rgb, predictor, "mser_fast", old)
            trial = process_frame(rgb, predictor, "mser_direct", new)
            if control["detections"] != trial["detections"]:
                raise ValueError("Direct-pointer full detections changed")
            deltas.append(check_predictions(trial["detections"], reference["detections"]))
        results[name] = dict(
            model_sha256=sha256(base / model["path"]),
            max_golden_score_delta=max(deltas),
            frames=len(deltas),
        )
    return dict(
        bundle_sha256=sha256(base / "bundle.json"),
        native_receipt=read_json(Path(library).with_suffix(".json")),
        script_sha256=sha256(__file__),
        results=results,
        exact_proposals=True,
        exact_full_detections=True,
        scope="Host parity only; no hardware speed or flight qualification",
        test_evaluated=False,
        deployment_approved=False,
        pi_timing_measured=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "library", "output"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    report = check(args.base, args.library)
    write_json(args.output, report)
    print(report["results"])
