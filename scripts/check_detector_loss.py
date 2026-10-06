"""Compare CPU/MPS detector loss before training; uses one labeled TRAIN image only."""

import argparse
from copy import deepcopy
import os
from pathlib import Path

from dtr.data import read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    config_dir = Path("artifacts/detector-settings").resolve()
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(config_dir)
    os.environ["YOLO_OFFLINE"] = "true"
    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLO, settings
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.cfg import get_cfg
    settings.update({"sync": False, "hub": False})
    root = Path("data/goal-detector-v10-20261005")
    receipt = read_json(root / "receipt.json")
    for row in receipt["splits"]["train"]["images"]:
        labels_path = root / "labels/train" / (Path(row["file"]).stem + ".txt")
        if labels_path.stat().st_size:
            break
    rgb = cv2.cvtColor(cv2.imread(str(root / "images/train" / row["file"])), cv2.COLOR_BGR2RGB)
    labels = np.loadtxt(labels_path).reshape(-1, 5)
    records = []
    for deterministic in (False, True):
        torch.manual_seed(42)
        torch.use_deterministic_algorithms(deterministic, warn_only=True)
        base = DetectionModel("yolo11n.yaml", nc=6, verbose=False)
        base.load(YOLO("artifacts/pretrained/yolo11n.pt").model, verbose=False)
        base.args = get_cfg()
        for device in ("cpu", "mps"):
            model = deepcopy(base).to(device).train()
            batch = dict(
                img=torch.tensor(rgb.copy().transpose(2, 0, 1), device=device)
                    .unsqueeze(0).repeat(2, 1, 1, 1).float()/255,
                batch_idx=torch.arange(2, device=device).repeat_interleave(len(labels)),
                cls=torch.tensor(labels[:, :1], device=device, dtype=torch.float32).repeat(2, 1),
                bboxes=torch.tensor(labels[:, 1:], device=device, dtype=torch.float32).repeat(2, 1),
            )
            with torch.no_grad():
                loss = model(batch)[1].cpu().tolist()
            record = dict(device=device, deterministic=deterministic, loss=loss)
            records.append(record)
            print(record, flush=True)
    write_json(output, dict(torch=torch.__version__, file=row["file"], split="train", records=records))


if __name__ == "__main__":
    main()
