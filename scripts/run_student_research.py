"""Bounded 2x2 architecture/loss experiment. Does not select a deployment model."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from dtr.data import read_json, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=25)
    args = parser.parse_args()
    if not 1 <= args.epochs <= 25:
        parser.error("Require 1..25 epochs")
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(root / "src"), "TF_CPP_MIN_LOG_LEVEL": "3",
           "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "4"}
    plan = [dict(name=f"{arch}-{loss}", architecture=arch, alpha=alpha)
            for arch in ("tiny", "context") for loss, alpha in (("kd", .5), ("supervised", 1.0))]
    sources = ["src/dtr/models.py", "scripts/distill_pi_student.py", "scripts/export_pi_float.py",
               "scripts/pi_optimize_bench.py", "scripts/run_student_research.py"]
    hashes = {}
    for name in sources:
        dest = out / "source" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, dest)
        hashes[name] = sha256(root / name)
    write_json(out / "plan.json", dict(
        experiments=plan, epochs=args.epochs, seed=42, source_sha256=hashes,
        purpose="controlled development ablation; not final test or deployment",
        test_evaluated=False, deployment_approved=False,
        comparison="Same data, seed, Adam 1e-3, no augmentation, best validation hard loss",
        limitations="One seed; reused correlated development scenes. No Pi timings in Mac scores."))
    results = []
    try:
        for item in plan:
            name = item["name"]
            run = out / name
            steps = [
                ("train", ["scripts/distill_pi_student.py", "--config", "configs/goal-proposals.json",
                           "--manifest", "data/dtr-proposals-reviewed-20261005/goal/manifest.json",
                           "--teacher", "runs/goal-proposals-20261005/teacher.keras",
                           "--target-cache-run", "runs/goal-pi-student-long-20261005",
                           "--output", str(run), "--epochs", str(args.epochs),
                           "--student-variant", item["architecture"], "--alpha", str(item["alpha"])]),
                ("float", ["scripts/export_pi_float.py", "--run", str(run),
                           "--output", str(run / "student.float.tflite")]),
                ("frames", ["scripts/pi_optimize_bench.py", "--base", "runs/pi-comparison-20261005",
                            "--model", str(run / "student.float.tflite"), "--mode", "frames",
                            "--output", str(run / "frames-mac.json")]),
            ]
            for stage, command in steps:
                print(f"{name}: {stage}", flush=True)
                write_json(out / "status.json", dict(state="running", experiment=name, stage=stage))
                with (out / f"{name}-{stage}.log").open("x") as log:
                    subprocess.run([sys.executable, *command], cwd=root, env=env,
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
            report = read_json(run / "report.json")
            frame = read_json(run / "frames-mac.json")
            results.append(dict(**item, parameters=report["student_parameters"],
                                crop=report["validation_fp32"], frames=frame))
            write_json(out / "results.json", dict(experiments=results, test_evaluated=False,
                                                deployment_approved=False))
            print(name, "crop accuracy", report["validation_fp32"]["accuracy"],
                  "yellow F1", frame["yellow_localization"]["f1"], flush=True)
        write_json(out / "status.json", dict(state="complete", completed_at=time.time()))
    except BaseException as exc:
        write_json(out / "status.json", dict(state="failed", error=str(exc)))
        raise


if __name__ == "__main__":
    main()
