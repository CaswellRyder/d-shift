"""Self-contained Pi bundle for the green/purple balloon student.

Capture targets are green/purple balloons; red/blue are the drones (2026-10-12).
The viewer runs the OpenCV ``balloon_green_purple`` search and the FP32 crop
student; the Pi checks the host-made goldens before any camera opens. Golden
frames come from the DTR validation set, which has competition lighting.
Research bundle only: no deployment or flight qualification.
"""

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image

from dtr.data import sha256, write_json
from dtr.runtime import Predictor

SCRIPTS = ("pi_balloon_web.py", "pi_balloon_web.html", "pi_balloon_live_bench.py",
           "pi_balloon_schedule_bench.py", "pi_balloon_search_bench.py", "pi_red_blue_bench.py",
           "research_pixel_proposals.py", "research_balloon_search.py",
           "research_balloon_search_fast.py")


def golden_frames(coco_root, count):
    """Evenly spaced validation frames that contain at least one green/purple balloon."""
    doc = json.loads((Path(coco_root) / "valid/_annotations.coco.json").read_text())
    targets = {c["id"] for c in doc["categories"] if c["name"] in ("Green Balloon", "Purple Balloon")}
    with_targets = sorted({a["image_id"] for a in doc["annotations"] if a["category_id"] in targets})
    names = {i["id"]: i["file_name"] for i in doc["images"]}
    step = max(1, len(with_targets) // count)
    return [Path(coco_root) / "valid" / names[i] for i in with_targets[::step][:count]]


def build(output, model, coco_root, alias, limit, count):
    from scripts.pi_balloon_web import GREEN_PURPLE_GOLDEN, green_purple_golden, prepare

    output, model = Path(output), Path(model)
    if output.exists():
        raise FileExistsError(output)
    for source in Path("src/dtr").glob("*.py"):
        (output / "dtr").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, output / "dtr" / source.name)
    for name in SCRIPTS:
        shutil.copyfile(Path("scripts") / name, output / name)
    relative = f"models/{alias}.tflite"
    (output / "models").mkdir()
    shutil.copyfile(model, output / relative)
    shutil.copyfile(model.with_suffix(".json"), (output / relative).with_suffix(".json"))
    predictor = Predictor(output / relative, allow_unvalidated=True)
    if predictor.metadata.get("tensor_dtype") != "float32":
        raise ValueError("The Pi bundles use FP32 students")
    frames = []
    (output / "frames").mkdir()
    for index, source in enumerate(golden_frames(coco_root, count)):
        path = f"frames/{index:03d}.png"
        Image.open(source).convert("RGB").resize((320, 240), Image.Resampling.BILINEAR).save(output / path)
        frames.append(dict(path=path, source=source.name))
    write_json(output / "inputs.json", dict(models={alias: dict(path=relative)}, default_model=alias,
                                            frames=frames, crops=[]))
    write_json(output / GREEN_PURPLE_GOLDEN,
               green_purple_golden(output, alias, output / relative, predictor, limit, frames))
    write_json(output / "bundle.json", dict(
        files={str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*")) if p.is_file()},
        default_model=alias, default_model_sha256=sha256(output / relative),
        builder_sha256=sha256(__file__), deployment_approved=False, test_evaluated=False,
        scope="Green/purple research student for the Pi; not flight qualification"))
    _, info, _ = prepare(output)
    return dict(path=str(output), bundle_sha256=sha256(output / "bundle.json"),
                golden_frames=info["golden_frames"], max_golden_score_delta=info["max_golden_score_delta"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True, help="FP32 student .tflite with its .json")
    parser.add_argument("--coco", default="data/roboflow-dtr-v10-grouped/coco")
    parser.add_argument("--alias", default="green-purple-context")
    parser.add_argument("--limit", type=int, default=8, help="proposals classified per frame")
    parser.add_argument("--frames", type=int, default=12)
    args = parser.parse_args()
    print(build(args.output, args.model, args.coco, args.alias, args.limit, args.frames))
