"""Build fresh current-student + full-scene parity bundle, never a deployment."""

import argparse
from pathlib import Path
import shutil

import cv2
from PIL import Image

from dtr.data import read_json, sha256, write_json
from dtr.runtime import Predictor
from scripts.build_red_blue_pi_bundle import build
from scripts.evaluate_reviewed_balloon_scenes import reviewed_frames
from scripts.pi_balloon_search_bench import process_frame


def prepare(output):
    output = Path(output)
    build(
        output,
        "data/balloon-red-blue-bootstrap-20261008/manifest.json",
        "configs/balloon-red-blue-development-scenes.json",
        "data/raw/matterport-balloon/balloon_dataset.zip",
        {
            f"new-views-{seed}": f"runs/balloon-red-blue-new-views-fixed-teacher{suffix}-20261008/student.fp32.tflite"
            for seed, suffix in ((42, ""), (43, "-seed43"))
        },
    )
    for name in ("pi_balloon_search_bench", "research_balloon_search"):
        shutil.copyfile(f"scripts/{name}.py", output / f"{name}.py")
    inputs = read_json(output / "inputs.json")
    for frame in inputs["frames"]:
        frame["panel"] = "original"
    frames, _ = reviewed_frames(
        Path("data/engdes2-development-scenes-20261008"),
        "configs/engdes2-development-scenes-20261008.json",
    )
    for i, (record, rgb) in enumerate(frames):
        relative = f"frames/indoor-{i:03d}.png"
        Image.fromarray(rgb).save(output / relative)
        inputs["frames"].append(
            dict(
                path=relative,
                source=record["source"],
                panel="indoor",
                truth=record["provisional_truth"],
            )
        )
    write_json(output / "inputs.json", inputs)
    cv2.setNumThreads(1)
    import numpy as np

    images = [np.asarray(Image.open(output / r["path"]).convert("RGB")) for r in inputs["frames"]]
    golden = {}
    for name, model in inputs["models"].items():
        predictor = Predictor(output / model["path"], allow_unvalidated=True)
        golden[name] = {
            search: [
                dict(
                    source=r["source"],
                    detections=process_frame(rgb, predictor, search)["detections"],
                )
                for rgb, r in zip(images, inputs["frames"])
            ]
            for search in ("baseline", "mser")
        }
    write_json(output / "search-golden.json", golden)
    receipt = dict(
        files={
            str(p.relative_to(output)): sha256(p)
            for p in output.rglob("*")
            if p.is_file() and p.name != "bundle.json"
        },
        deployment_approved=False,
        test_evaluated=False,
        scope="Current red/blue student replay research; no camera or actuation",
        builder_sha256=sha256(__file__),
        original_review_sha256=sha256("configs/balloon-red-blue-development-scenes.json"),
        indoor_review_sha256=sha256("configs/engdes2-development-scenes-20261008.json"),
    )
    write_json(output / "bundle.json", receipt)
    return dict(
        path=str(output),
        frames=len(images),
        models=list(inputs["models"]),
        bundle_sha256=sha256(output / "bundle.json"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(prepare(args.output))
