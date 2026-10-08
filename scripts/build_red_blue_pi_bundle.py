"""Build an isolated, hash-bound red/blue research bundle; never includes test images."""
import argparse
import io
from pathlib import Path
import shutil
import zipfile

import cv2
import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, write_json
from dtr.runtime import Predictor
from dtr.vision import RED_BLUE_PROFILE, validate_model_profile


def build(output, manifest, review, archive_path, models, red_blue_lookup=None):
    output, manifest = Path(output), Path(manifest)
    doc, decisions = read_json(manifest), read_json(review)
    if sha256(manifest) != decisions["manifest_sha256"]:
        raise ValueError("Development review manifest changed")
    if sha256(archive_path) != doc["source_archive_sha256"]:
        raise ValueError("Source archive changed")
    rows = [r for r in doc["samples"] if r["split"] == "val"]
    if not rows or {r["source_image"] for r in rows} != set(decisions["sources"]):
        raise ValueError("Development scene review incomplete")
    output.mkdir(parents=True, exist_ok=False)
    (output / "dtr").mkdir()
    (output / "models").mkdir()
    (output / "crops").mkdir()
    (output / "frames").mkdir()
    for name in ("__init__", "data", "runtime", "native_tflite", "goal_evidence",
                 "vision", "tracking", "temporal", "color_lookup"):
        shutil.copyfile(f"src/dtr/{name}.py", output / "dtr" / f"{name}.py")
    shutil.copyfile("scripts/pi_red_blue_bench.py", output / "pi_red_blue_bench.py")
    if red_blue_lookup:
        from dtr.color_lookup import RedBlueColorLookup
        RedBlueColorLookup(red_blue_lookup)  # Verify before copying.
        shutil.copyfile(red_blue_lookup, output / "red-blue-lookup.bin")
        shutil.copyfile(Path(red_blue_lookup).with_suffix(".json"), output / "red-blue-lookup.json")
        for name in ("build_red_blue_lookup", "pi_red_blue_search_bench", "build_native_color_lookup"):
            shutil.copyfile(f"scripts/{name}.py", output / f"{name}.py")
        for name in ("native_color_lookup.py", "native_color_lookup.c"):
            shutil.copyfile(f"src/dtr/{name}", output / "dtr" / name)
    for name in ("LICENSE", "THIRD_PARTY_NOTICES.md"):
        shutil.copyfile(name, output / name)
    crops = []
    for i, row in enumerate(rows):
        source = (manifest.parent / row["path"]).resolve()
        if not source.is_relative_to(manifest.parent.resolve()) or sha256(source) != row["sha256"]:
            raise ValueError("Development crop changed or escaped dataset")
        relative = f"crops/{i:03}.png"
        with Image.open(source) as image:
            image.convert("RGB").save(output / relative)
        crops.append(dict(path=relative, truth=row["label"]))
    model_records = {}
    for name, source in models.items():
        if not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in name):
            raise ValueError("Invalid model alias")
        source = Path(source)
        relative = f"models/{name}.tflite"
        shutil.copyfile(source, output / relative)
        shutil.copyfile(source.with_suffix(".json"), (output / relative).with_suffix(".json"))
        predictor = Predictor(output / relative, allow_unvalidated=True)
        validate_model_profile(predictor.metadata, RED_BLUE_PROFILE)
        golden = []
        for row in crops:
            pred = predictor.predict(output / row["path"])
            golden.append(dict(path=row["path"], scores=pred["scores"], label=pred["label"],
                               accepted=pred["accepted"]))
        model_records[name] = dict(path=relative, golden=golden,
                                  score_tolerance=.02 if predictor.metadata.get("tensor_dtype", "int8") == "int8" else .001)
    frames = []
    with zipfile.ZipFile(archive_path) as archive:
        for i, source in enumerate(sorted(decisions["sources"])):
            targets = [r for r in rows if r["source_image"] == source and r["label"] != "background"]
            if sorted(r["annotation_id"] for r in targets) != sorted(decisions["sources"][source]["target_ids"]):
                raise ValueError("Reviewed targets changed")
            with Image.open(io.BytesIO(archive.read(source))) as image:
                width, height = image.size
                rgb = cv2.resize(np.asarray(image.convert("RGB")), (320, 240), interpolation=cv2.INTER_AREA)
            relative = f"frames/{i:03}.png"
            Image.fromarray(rgb).save(output / relative)
            truth = [dict(label=r["label"], box=[v*(320/width if j % 2 == 0 else 240/height)
                                               for j, v in enumerate(r["box_xyxy"])]) for r in targets]
            frames.append(dict(path=relative, source=source, truth=truth))
    write_json(output / "inputs.json", dict(split="development_validation", crops=crops, frames=frames,
                                           models=model_records, manifest_sha256=sha256(manifest)))
    files = {str(p.relative_to(output)): sha256(p) for p in output.rglob("*") if p.is_file()}
    write_json(output / "bundle.json", dict(files=files, deployment_approved=False, test_evaluated=False,
               scope="Private off-domain development replay and unlabeled Pi timing; not flight qualification"))
    return dict(bundle=str(output), models=list(models), crops=len(crops), frames=len(frames))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--review", default="configs/balloon-red-blue-development-scenes.json")
    parser.add_argument("--archive", default="data/raw/matterport-balloon/balloon_dataset.zip")
    parser.add_argument("--model", action="append", required=True, help="Unique alias=path")
    parser.add_argument("--red-blue-lookup", help="Optional prebuilt exact RGB24 color table")
    args = parser.parse_args()
    models = dict(item.split("=", 1) for item in args.model)
    if len(models) != len(args.model):
        parser.error("Duplicate model alias")
    print(build(args.output, args.manifest, args.review, args.archive, models, args.red_blue_lookup))


if __name__ == "__main__":
    main()
