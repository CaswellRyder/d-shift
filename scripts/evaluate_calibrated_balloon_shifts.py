"""Fixed photometric stress check; transformed development photos are not new scenes."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, write_json
from dtr.runtime import Predictor
from scripts.calibrate_balloon_split_geometry import calibration_bound
from scripts.compare_calibrated_balloon_selection import raw_candidates
from scripts.evaluate_red_blue_development import LABELS, detection_counts, metrics
from scripts.pi_balloon_search_bench import check_predictions, golden_key, process_frame, select_detections
from scripts.pi_red_blue_bench import verify_bundle
from scripts.research_pixel_photometric import TRANSFORMS, photometric


def evaluate(base, output):
    base, output = Path(base), Path(output)
    if output.exists():
        raise FileExistsError(output)
    receipt = verify_bundle(base)
    inputs = read_json(base/"inputs.json")
    calibration = read_json(base/"split-calibration.json")
    if (receipt["calibration_sha256"] != sha256(base/"split-calibration.json")
            or calibration.get("training_only") is not True
            or calibration.get("test_evaluated") is not False
            or calibration.get("deployment_approved") is not False):
        raise ValueError("Require bound training-only calibration")
    aspect = calibration_bound(calibration)
    if (not inputs.get("split", "").startswith(("development", "lockbox_panel"))
            or receipt.get("test_evaluated") is not False
            or receipt.get("deployment_approved") is not False):
        raise ValueError("Require evaluation-only development bundle")
    model = inputs["default_model"]
    path = base/inputs["models"][model]["path"]
    predictor = Predictor(path, allow_unvalidated=True)
    if predictor.metadata["threshold"] != .8 or predictor.metadata["classes"] != ["background", *LABELS]:
        raise ValueError("Classifier operating point changed")
    golden = read_json(base/"search-golden.json")[model]
    images = [np.asarray(Image.open(base/f["path"]).convert("RGB")) for f in inputs["frames"]]
    results = []
    for transform, gain, gamma in TRANSFORMS:
        panels, frames = {}, []
        for index, (f, rgb) in enumerate(zip(inputs["frames"], images, strict=True)):
            shifted = photometric(rgb, gain, gamma)
            result = process_frame(shifted, predictor, "mser", selection="split_merged")
            original = result["detections"]
            changed = select_detections(raw_candidates(original), "split_merged", aspect)
            if transform == "identity":
                for bound, found in ((1.5, original), (aspect, changed)):
                    check_predictions(found, golden[golden_key("mser", "split_merged", bound)][index]["detections"])
            entry = dict(index=index, source=f["source"], panel=f["panel"], counts={})
            for policy, found in (("control", original), ("calibrated", changed)):
                counts = detection_counts(f["truth"], found)
                entry["counts"][policy] = counts
                totals = panels.setdefault(f["panel"], {}).setdefault(
                    policy, {label: dict(tp=0, fp=0, fn=0) for label in LABELS})
                for label in LABELS:
                    for key in totals[label]:
                        totals[label][key] += counts[label][key]
            frames.append(entry)
        panel_metrics = {p:{k:metrics(v) for k,v in policies.items()} for p,policies in panels.items()}
        results.append(dict(transform=transform, gain=list(gain), gamma=gamma, panels=panel_metrics, frames=frames))
        print(transform, panel_metrics, flush=True)
    report = dict(scope="Synthetic photometric development stress, not real lighting or qualification",
                  model_sha256=sha256(path), bundle_sha256=sha256(base/"bundle.json"),
                  calibration_sha256=sha256(base/"split-calibration.json"), script_sha256=sha256(__file__),
                  transform_script_sha256=sha256("scripts/research_pixel_photometric.py"),
                  benchmark_script_sha256=sha256("scripts/pi_balloon_search_bench.py"),
                  unique_source_images=len(images), transforms=len(TRANSFORMS), results=results,
                  test_evaluated=False, deployment_approved=False, pi_timing_measured=False,
                  independent_recording_sessions_verified=False)
    write_json(output, report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    evaluate(a.base, a.output)
