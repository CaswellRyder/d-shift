"""Run both task pipelines in new directories; never overwrite an older experiment."""

import datetime
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAMP = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def call(*args):
    subprocess.run([sys.executable, "-m", "dtr.cli", *args], cwd=ROOT, check=True)


if __name__ == "__main__":
    pretrained = ROOT / "artifacts/pretrained/mobilenetv4_conv_small.keras"
    if not pretrained.exists():
        call("--cpu", "pretrained", "--output", str(pretrained))
    for task in ("balloon", "goal"):
        config = f"configs/{task}.json"
        data = f"data/smoke-{task}-{STAMP}"
        call("synthetic", "--config", config, "--output", data)
        call(
            "train",
            "--config",
            config,
            "--manifest",
            f"{data}/manifest.json",
            "--pretrained",
            str(pretrained),
            "--output",
            f"runs/{task}-smoke-{STAMP}",
            "--smoke",
        )
