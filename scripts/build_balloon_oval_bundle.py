"""Build isolated baseline/calibrated Pi goldens without changing active deployment."""
import argparse
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, write_json
from dtr.native_balloon_regions import NativeBalloonRegions
from dtr.runtime import Predictor
from scripts.calibrate_balloon_split_geometry import calibration_bound
from scripts.pi_balloon_search_bench import check_predictions, golden_key, process_frame
from scripts.pi_red_blue_bench import verify_bundle


def prepare(reference, calibration_path, output, library):
    reference, output = Path(reference), Path(output)
    if output.exists():
        raise FileExistsError(output)
    receipt = verify_bundle(reference)
    inputs = read_json(reference/"inputs.json")
    calibration = read_json(calibration_path)
    if (calibration.get("schema") != "balloon-split-geometry-v1"
            or calibration.get("training_only") is not True
            or calibration.get("test_evaluated") is not False
            or calibration.get("deployment_approved") is not False):
        raise ValueError("Require training-only calibration")
    bound = calibration_bound(calibration)
    if (not inputs.get("split", "").startswith(("development", "lockbox_panel"))
            or receipt.get("test_evaluated") is not False
            or receipt.get("deployment_approved") is not False):
        raise ValueError("Require an evaluation-only development reference bundle")
    golden_key("mser", "split_merged", bound)  # Validate before writing.
    name = inputs["default_model"]
    for relative in receipt["files"]:
        dest = output/relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(reference/relative, dest)
    for filename in ("pi_balloon_search_bench.py", "research_balloon_search.py"):
        shutil.copyfile(Path("scripts")/filename, output/filename)
    shutil.copyfile(calibration_path, output/"split-calibration.json")
    old_goldens = read_json(reference/"search-golden.json")[name]["mser_split_merged"]
    predictor = Predictor(output/inputs["models"][name]["path"], allow_unvalidated=True)
    native = NativeBalloonRegions(library, direct=True)
    golden = {name: {golden_key("mser", "split_merged", v): [] for v in (1.5, bound)}}
    for row, previous in zip(inputs["frames"], old_goldens, strict=True):
        with Image.open(output/row["path"]) as im:
            rgb = np.asarray(im.convert("RGB"))
        for aspect in (1.5, bound):
            plain = process_frame(rgb, predictor, "mser", selection="split_merged", split_max_aspect=aspect)
            fast = process_frame(rgb, predictor, "mser_direct", native, "split_merged", aspect)
            check_predictions(plain["detections"], fast["detections"])
            if aspect == 1.5:
                check_predictions(plain["detections"], previous["detections"])
            golden[name][golden_key("mser", "split_merged", aspect)].append(
                dict(source=row["source"], detections=plain["detections"]))
    write_json(output/"search-golden.json", golden)
    write_json(output/"bundle.json", dict(
        files={str(p.relative_to(output)): sha256(p) for p in output.rglob("*")
               if p.is_file() and p != output/"bundle.json"},
        reference_bundle_sha256=sha256(reference/"bundle.json"),
        calibration_sha256=sha256(calibration_path), builder_sha256=sha256(__file__),
        default_model=name, aspect_bounds=[1.5, bound],
        deployment_approved=False, test_evaluated=False,
        scope="Isolated fixed-weight research replay; policy-specific goldens, no camera or actuation"))
    verify_bundle(output)
    return dict(path=str(output), frames=len(inputs["frames"]), model=name,
                bundle_sha256=sha256(output/"bundle.json"))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "calibration", "output", "library"):
        p.add_argument(f"--{name}", required=True)
    a = p.parse_args()
    print(prepare(a.reference, a.calibration, a.output, a.library))
