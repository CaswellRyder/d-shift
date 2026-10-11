"""Add a replacement student to a verified Pi bundle and make it the default model.

Existing weights, inputs and goldens are copied unchanged; the old students stay
available by name. Research bundle only: no deployment or flight qualification.
"""

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

LIVE_SCRIPTS = ("pi_balloon_live_bench.py", "pi_balloon_live_schedule_bench.py",
                "pi_balloon_web.py", "pi_balloon_web.html")


def prepare(reference, output, library, name, model):
    reference, output, model = Path(reference), Path(output), Path(model)
    if output.exists():
        raise FileExistsError(output)
    if not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in name):
        raise ValueError("Invalid model alias")
    verify_bundle(reference)
    inputs, golden = read_json(reference / "inputs.json"), read_json(reference / "search-golden.json")
    if name in inputs["models"] or name in golden:
        raise ValueError("Model alias already in bundle")
    for relative in read_json(reference / "bundle.json")["files"]:
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(reference / relative, target)
    for script in LIVE_SCRIPTS:
        shutil.copyfile(Path("scripts") / script, output / script)
    relative = f"models/{name}.tflite"
    shutil.copyfile(model, output / relative)
    shutil.copyfile(model.with_suffix(".json"), (output / relative).with_suffix(".json"))
    predictor = Predictor(output / relative, allow_unvalidated=True)
    if predictor.metadata.get("tensor_dtype", "int8") == "int8":
        raise ValueError("Expected an FP32 student, like the bundled ones")
    crops = []
    for row in inputs["crops"]:
        pred = predictor.predict(output / row["path"])
        crops.append(dict(path=row["path"], scores=pred["scores"], label=pred["label"],
                          accepted=pred["accepted"]))
    inputs["models"][name] = dict(path=relative, golden=crops, score_tolerance=.001)
    inputs["default_model"] = name
    native = NativeBalloonRegions(library, direct=True)
    images = [np.asarray(Image.open(output / r["path"]).convert("RGB")) for r in inputs["frames"]]
    golden[name] = {search: [] for search in ("baseline", "mser", "mser_bright")}
    for rgb, record in zip(images, inputs["frames"], strict=True):
        for search in golden[name]:
            detections = process_frame(rgb, predictor, search, native if search == "mser_bright" else None)[
                "detections"]
            golden[name][search].append(dict(source=record["source"], detections=detections))
        # The Pi runs the native direct search; it must reproduce the exact search here.
        check_predictions(process_frame(rgb, predictor, "mser_direct", native)["detections"],
                          golden[name]["mser"][-1]["detections"])
    write_json(output / "inputs.json", inputs)
    write_json(output / "search-golden.json", golden)
    manifest = dict(
        files={str(p.relative_to(output)): sha256(p) for p in output.rglob("*") if p.is_file()},
        reference_bundle_sha256=sha256(reference / "bundle.json"),
        default_model=name,
        default_model_sha256=sha256(output / relative),
        builder_sha256=sha256(__file__),
        deployment_approved=False,
        test_evaluated=False,
        scope="Replacement research student for the Pi; not flight qualification",
    )
    write_json(output / "bundle.json", manifest)
    verify_bundle(output)
    return dict(path=str(output), bundle_sha256=sha256(output / "bundle.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("reference", "output", "library", "name", "model"):
        parser.add_argument(f"--{arg}", required=True)
    args = parser.parse_args()
    print(prepare(args.reference, args.output, args.library, args.name, args.model))
