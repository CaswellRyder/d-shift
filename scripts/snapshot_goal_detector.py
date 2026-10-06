"""Copy a stable saved last checkpoint for development diagnostics, never promotion."""

import argparse
import hashlib
import os
from pathlib import Path

from dtr.data import read_json, sha256, write_json


def stable_bytes(path):
    """Fail if a writer replaced/truncated the checkpoint during the read."""
    blob = path.read_bytes()
    if not blob or hashlib.sha256(blob).hexdigest() != sha256(path):
        raise ValueError("Checkpoint changed during capture; retry after a completed epoch")
    return blob


def saved_epoch(checkpoint, completed, progress_epoch=None):
    if completed:
        epochs = checkpoint.get("train_results", {}).get("epoch", [])
        if (checkpoint["epoch"] != -1 or checkpoint.get("optimizer") is not None
                or checkpoint.get("ema") is not None or checkpoint.get("model") is None
                or not epochs or epochs[-1] != progress_epoch
                or progress_epoch < 1 or int(progress_epoch) != progress_epoch):
            raise ValueError("Finalized last checkpoint does not match recorded completed epoch")
        return int(progress_epoch)
    epoch = checkpoint["epoch"] + 1
    if epoch < 1:
        raise ValueError("No resumable completed epoch in checkpoint")
    return epoch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run, output = Path(args.run).resolve(), Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    experiment = read_json(run / "experiment.json")
    segment_boundary = (experiment["status"] == "interrupted"
                        and experiment.get("interruption_reason") == "process_epoch_limit")
    completed = experiment["status"] == "complete"
    if experiment["status"] != "running" and not segment_boundary and not completed:
        raise ValueError("Capture requires a running experiment, saved boundary or completed run")
    source = run / "fit/weights/last.pt"
    blob = stable_bytes(source)
    config_dir = Path("artifacts/detector-settings").resolve()
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(config_dir)
    os.environ["YOLO_OFFLINE"] = "true"
    import io
    import torch
    import yaml
    checkpoint = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=False)
    epoch = saved_epoch(checkpoint, completed,
                        read_json(run / "progress.json")["epoch"] if completed else None)
    model = checkpoint["model"] if completed else checkpoint["ema"]
    names = [model.names[i] for i in range(len(model.names))]
    standard = read_json("configs/goal-detection-standard.json")
    if names != standard["classes"]:
        raise ValueError("Wrong goal class order")
    train_args = yaml.safe_load((run / "fit/args.yaml").read_text())
    dataset_root = Path(train_args["data"]).parent
    if sha256(dataset_root / "receipt.json") != experiment["dataset_receipt_sha256"]:
        raise ValueError("Dataset receipt changed")
    if hashlib.sha256(blob).hexdigest() != sha256(source):
        raise ValueError("Checkpoint changed during inspection")
    output.mkdir(parents=True)
    (output / "model.pt").write_bytes(blob)
    write_json(output / "snapshot.json", dict(
        kind="interim_checkpoint_snapshot", epoch=epoch, parent_run=str(run),
        model_sha256=sha256(output / "model.pt"), experiment=experiment,
        dataset_root=str(dataset_root), deployment_approved=False, promotion_allowed=False,
        test_evaluated=False, finalized_last=completed,
    ))
    print(dict(output=str(output), epoch=epoch, promotion_allowed=False))


if __name__ == "__main__":
    main()
