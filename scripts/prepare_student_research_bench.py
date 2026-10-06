"""Freeze a small transferable model/crop benchmark; never connects to the Pi."""
import argparse
from pathlib import Path
import shutil

from dtr.data import read_json, sha256, write_json
from dtr.runtime import Predictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True, help="name=model.tflite")
    parser.add_argument("--base", default="runs/pi-comparison-20261005")
    parser.add_argument("--output", required=True)
    parser.add_argument("--include-replay", action="store_true",
                        help="Also snapshot temporal replay code; frozen base images remain separate")
    parser.add_argument("--color-lookup", help="Optional prebuilt goal RGB lookup for isolated replay experiments")
    args = parser.parse_args()
    output, base = Path(args.output), Path(args.base)
    output.mkdir(parents=True, exist_ok=False)
    for name in ("LICENSE", "THIRD_PARTY_NOTICES.md"):
        shutil.copyfile(name, output / name)
    (output / "dtr").mkdir()
    modules = ["__init__", "data", "runtime", "native_tflite", "goal_evidence"]
    if args.include_replay:
        modules += ["vision", "tracking", "temporal", "goal_search", "color_lookup"]
        shutil.copyfile("scripts/pi_temporal_bench.py", output / "pi_temporal_bench.py")
        shutil.copyfile("scripts/benchmark_tracking_backends.py", output / "benchmark_tracking_backends.py")
    if args.color_lookup:
        if not args.include_replay:
            parser.error("Color lookup needs --include-replay")
        path = Path(args.color_lookup)
        shutil.copyfile(path, output / "goal-colors.bin")
        shutil.copyfile(path.with_suffix(".json"), output / "goal-colors.json")
        shutil.copyfile("scripts/build_goal_lookup.py", output / "build_goal_lookup.py")
        shutil.copyfile("scripts/benchmark_goal_lookup.py", output / "benchmark_goal_lookup.py")
    for name in modules:
        shutil.copyfile(f"src/dtr/{name}.py", output / "dtr" / f"{name}.py")
    shutil.copyfile("scripts/pi_research_crops.py", output / "pi_research_crops.py")
    predictors, models = {}, {}
    (output / "models").mkdir()
    for item in args.model:
        name, path = item.split("=",1)
        if not name or not all(c.isalnum() or c in "-_" for c in name) or name in models:
            raise ValueError("Model name must be unique and alphanumeric with -/_")
        source = Path(path)
        relative = f"models/{name}.tflite"
        shutil.copyfile(source, output / relative)
        shutil.copyfile(source.with_suffix(".json"), (output / relative).with_suffix(".json"))
        predictors[name] = Predictor(output / relative, True)
        if predictors[name].metadata["task"] != "goal":
            raise ValueError("Goal comparison only")
        models[name] = relative
    crops = []
    (output / "golden").mkdir()
    for row in read_json(base / "golden.json"):
        if row["task"] != "goal":
            continue
        source = (base / row["path"]).resolve()
        if not source.is_relative_to(base.resolve()) or sha256(source) != row["sha256"]:
            raise ValueError("Frozen golden crop mismatch")
        shutil.copyfile(source, output / row["path"])
        reference = {name: predictor.predict(source) for name,predictor in predictors.items()}
        crops.append(dict(path=row["path"], sha256=row["sha256"], reference=reference))
    write_json(output / "golden.json", dict(models=models, crops=crops))
    hashes = {str(p.relative_to(output)): sha256(p) for p in output.rglob("*") if p.is_file()}
    base_hashes = ({name: sha256(base / name) for name in ("frames.json", "pi_compare.py", "baseline.py")}
                   if args.include_replay else {})
    write_json(output / "bundle.json", dict(files=hashes, replay_base_sha256=base_hashes,
                                           deployment_approved=False,
                                           purpose="observational crop/parity and optional synthetic replay research"))
    print(dict(output=str(output), models=len(models), crops=len(crops), files=len(hashes)))


if __name__ == "__main__":
    main()
