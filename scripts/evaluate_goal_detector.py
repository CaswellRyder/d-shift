"""Fixed-confidence grouped validation of completed runs or explicitly interim snapshots."""

import argparse
from collections import defaultdict
import os
from pathlib import Path
import yaml

from dtr.data import read_json, write_json, sha256
from dtr.detector_metrics import summarize
from dtr.detector_nms import offline_predictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--run")
    source_group.add_argument("--snapshot", help="Immutable interim snapshot; cannot approve promotion")
    parser.add_argument("--output", required=True)
    parser.add_argument("--input-width", type=int, choices=[320, 640, 960], default=640,
                        help="960 is an offboard upsampling experiment, not added source detail or Pi proof")
    parser.add_argument("--device", choices=["cpu", "mps"], default="mps")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    interim = args.snapshot is not None
    if interim:
        snapshot = Path(args.snapshot)
        receipt = read_json(snapshot / "snapshot.json")
        if receipt["kind"] != "interim_checkpoint_snapshot" or receipt["promotion_allowed"]:
            raise ValueError("Invalid interim snapshot receipt")
        experiment = receipt["experiment"]
        model_path = snapshot / "model.pt"
        expected = receipt["model_sha256"]
        dataset_root = Path(receipt["dataset_root"])
    else:
        run = Path(args.run)
        experiment = read_json(run / "experiment.json")
        if experiment["status"] != "complete":
            raise ValueError("Require completed run or explicit immutable interim snapshot")
        model_path = run / "fit/weights/best.pt"
        expected = experiment["best_model_sha256"]
        dataset_root = Path(yaml.safe_load((run / "fit/args.yaml").read_text())["data"]).parent
    digest = sha256(model_path)
    if digest != expected:
        raise ValueError("Checkpoint checksum mismatch")
    dataset_receipt = read_json(dataset_root / "receipt.json")
    if sha256(dataset_root / "receipt.json") != experiment["dataset_receipt_sha256"]:
        raise ValueError("Dataset receipt changed")
    standard_path = Path("configs/goal-detection-standard.json")
    standard = read_json(standard_path)
    if sha256(standard_path) != experiment["standard_sha256"]:
        raise ValueError("Acceptance standard changed since training started")
    config_dir = Path("artifacts/detector-settings").resolve()
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(config_dir)
    os.environ["YOLO_OFFLINE"] = "true"
    from ultralytics import YOLO, settings
    import cv2
    import torch
    if args.device == "cpu":
        torch.set_num_threads(4)
    settings.update({"sync": False, "hub": False})
    model = YOLO(str(model_path))
    names = [model.names[i] for i in range(len(model.names))]
    if names != standard["classes"]:
        raise ValueError("Checkpoint goal classes differ")
    source = Path("data/roboflow-dtr-v10-grouped/coco/valid")
    annotation_path = source / "_annotations.coco.json"
    validation_receipt = dataset_receipt["splits"]["valid"]
    if sha256(annotation_path) != validation_receipt["annotation_sha256"]:
        raise ValueError("Validation annotations changed since preparation")
    doc = read_json(annotation_path)
    aliases = read_json("configs/goal.json")["aliases"]
    categories = {c["id"]: c["name"] for c in doc["categories"]}
    annotations = defaultdict(list)
    for ann in doc["annotations"]:
        annotations[ann["image_id"]].append(ann)
    frames = []
    expected_images = {im["file"]: im["sha256"] for im in validation_receipt["images"]}
    if set(expected_images) != {im["file_name"] for im in doc["images"]}:
        raise ValueError("Validation frame set changed")
    for index, im in enumerate(doc["images"]):
        if sha256(source / im["file_name"]) != expected_images[im["file_name"]]:
            raise ValueError("Validation image changed since preparation")
        bgr = cv2.imread(str(source / im["file_name"]))
        if bgr is None:
            raise ValueError(f"Unreadable image: {im['file_name']}")
        truth = []
        for ann in annotations[im["id"]]:
            label = aliases.get(categories[ann["category_id"]])
            if label:
                x, y, w, h = ann["bbox"]
                truth.append(dict(label=label, box=[x, y, x+w, y+h]))
        if args.input_width == 320:
            bgr = cv2.resize(bgr, (320, 240))
        result = model.predict(bgr, imgsz=args.input_width, conf=standard["confidence_threshold"],
                               iou=0.7, max_det=100, device=args.device, half=False, verbose=False,
                               predictor=offline_predictor())[0]
        sx, sy = im["width"]/bgr.shape[1], im["height"]/bgr.shape[0]
        predictions = []
        for box in result.boxes.cpu().data.tolist():
            x1, y1, x2, y2, score, cls = box
            predictions.append(dict(label=names[int(cls)], score=score,
                                    box=[x1*sx, y1*sy, x2*sx, y2*sy]))
        frames.append(dict(file=im["file_name"], image_size=[im["width"], im["height"]],
                           truth=truth, predictions=predictions))
        if index % 100 == 0:
            print(index, "of", len(doc["images"]), flush=True)
    report = dict(split="val", frames=frames, model_sha256=digest,
                  annotation_sha256=sha256(annotation_path), standard_sha256=sha256(standard_path),
                  standard=standard, detector_input_width=args.input_width,
                  device=args.device, interim=interim, promotion_allowed=False,
                  postprocessing="complete-cpu-nms-v2",
                  epoch=receipt["epoch"] if interim else None,
                  scope="Correlated development validation, not independent test or Pi performance",
                  metrics=summarize(frames, standard))
    if sha256(model_path) != digest:
        raise ValueError("Checkpoint changed during evaluation")
    write_json(output, report)
    print({k: v for k, v in report["metrics"].items() if k != "size_strata"})


if __name__ == "__main__":
    main()
