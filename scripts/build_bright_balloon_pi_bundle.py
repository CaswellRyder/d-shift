"""Build a changed-policy research bundle, preserving the original golden."""

import argparse
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, write_json
from dtr.native_balloon_regions import NativeBalloonRegions
from dtr.runtime import Predictor
from scripts.pi_balloon_search_bench import check_predictions, process_frame
from scripts.pi_red_blue_bench import verify_bundle


def prepare(reference, output, library):
    reference, output = Path(reference), Path(output)
    if output.exists():
        raise FileExistsError(output)
    verify_bundle(reference)
    old = read_json(reference / "bundle.json")
    for name in old["files"]:
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(reference / name, target)
    for name in ("pi_balloon_search_bench", "research_balloon_search_fast"):
        shutil.copyfile(f"scripts/{name}.py", output / f"{name}.py")
    inputs, golden = read_json(output / "inputs.json"), read_json(output / "search-golden.json")
    native = NativeBalloonRegions(library, direct=True)
    images = [np.asarray(Image.open(output / r["path"]).convert("RGB")) for r in inputs["frames"]]
    for name, model in inputs["models"].items():
        if "mser_bright" in golden[name]:
            raise ValueError("Bright-only golden already exists")
        predictor = Predictor(output / model["path"], allow_unvalidated=True)
        rows = []
        for rgb, expected in zip(images, golden[name]["mser"], strict=True):
            current = process_frame(rgb, predictor, "mser_direct", native)
            check_predictions(current["detections"], expected["detections"])
            trial = process_frame(rgb, predictor, "mser_bright", native)
            rows.append(dict(detections=trial["detections"]))
        golden[name]["mser_bright"] = rows
    write_json(output / "search-golden.json", golden)
    manifest = dict(
        files={str(p.relative_to(output)): sha256(p) for p in output.rglob("*") if p.is_file()},
        reference_bundle_sha256=sha256(reference / "bundle.json"),
        original_golden_sha256=sha256(reference / "search-golden.json"),
        builder_sha256=sha256(__file__),
        deployment_approved=False,
        test_evaluated=False,
        scope="Changed bright-only proposal policy, not exact-search equivalence or flight qualification",
    )
    write_json(output / "bundle.json", manifest)
    verify_bundle(output)
    return dict(path=str(output), bundle_sha256=sha256(output / "bundle.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "output", "library"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    print(prepare(args.reference, args.output, args.library))
