"""Train-only exposure changes, without new images, labels or validation leakage."""

from collections import Counter
from copy import deepcopy
import math
from pathlib import Path
import shutil

import yaml

from .data import read_json, sha256, write_json


def parse_rows(text, class_count):
    rows = []
    for line in text.splitlines():
        values = list(map(float, line.split()))
        if len(values) != 5 or not all(math.isfinite(v) for v in values):
            raise ValueError("Invalid YOLO label row")
        cls, x, y, w, h = values
        if (int(cls) != cls or not 0 <= cls < class_count or not 0 <= x <= 1
                or not 0 <= y <= 1 or not 0 < w <= 1 or not 0 < h <= 1):
            raise ValueError("Invalid normalized class or box")
        rows.append(values)
    return rows


def prepare_close_range(source, output, *, threshold=.25, copies=4):
    """Repeat every train frame with a large goal; copy all its original labels.

    Copies count includes the original. This increases exposure, not independent data.
    Selection uses training boxes alone, equally for all six classes.
    """
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Derived output must be separate from its source")
    if not math.isfinite(threshold) or not 0 < threshold < 1 or not 2 <= copies <= 8:
        raise ValueError("Require threshold in (0,1) and 2..8 total copies")
    if not isinstance(copies, int):
        raise ValueError("Copies must be an integer")
    receipt_path = source / "receipt.json"
    receipt_digest = sha256(receipt_path)
    original = read_json(receipt_path)
    if (original["test_exported"] or original["test_evaluated"]
            or original["deployment_approved"] or set(original["splits"]) != {"train", "valid"}
            or len(original["classes"]) != 6 or original.get("training_exposure")):
        raise ValueError("Require unapproved original six-class train/valid dataset")
    if sha256(source / "dataset.yaml") != original["dataset_yaml_sha256"]:
        raise ValueError("Source YAML changed")
    planned = {}
    split_hashes = {}
    selected, qualifying_counts = [], Counter()
    # Verify every source before creating an output. Never read a reserved-test directory.
    for split in ("train", "valid"):
        records = original["splits"][split]["images"]
        names = [r["file"] for r in records]
        if len(set(names)) != len(names) or len({Path(n).stem for n in names}) != len(names):
            raise ValueError("Repeated image or label filename")
        planned[split] = []
        split_hashes[split] = set()
        for record in records:
            name = record["file"]
            if Path(name).name != name or name.lower().startswith("highbay"):
                raise ValueError("Unsafe filename or reserved recording")
            image = source / "images" / split / name
            label = source / "labels" / split / (Path(name).stem + ".txt")
            if sha256(image) != record["sha256"] or sha256(label) != record["labels_sha256"]:
                raise ValueError("Source image or labels changed")
            rows = parse_rows(label.read_text(), len(original["classes"]))
            qualifying = [r for r in rows if max(r[3:]) > threshold]
            multiplicity = copies if split == "train" and qualifying else 1
            if multiplicity > 1:
                selected.append(name)
                qualifying_counts.update(original["classes"][int(r[0])] for r in qualifying)
            planned[split].append((record, rows, multiplicity))
            split_hashes[split].add(record["sha256"])
    if split_hashes["train"] & split_hashes["valid"]:
        raise ValueError("Cross-split duplicate image")
    if not selected:
        raise ValueError("No qualifying training frames")
    if sha256(receipt_path) != receipt_digest:
        raise ValueError("Source receipt changed")
    result = deepcopy(original)
    result["training_exposure"] = dict(
        policy="repeat_train_frames_with_any_goal_longest_side_above_fraction",
        threshold=threshold, total_copies=copies, selected_files=sorted(selected),
        qualifying_boxes_by_class=dict(qualifying_counts),
        parent_dataset=str(source), parent_receipt_sha256=receipt_digest,
        unique_training_frames=len(planned["train"]),
        scope="Exposure only; no new independent images, boxes, labels or validation changes",
    )
    output.mkdir(parents=True)
    for split, plans in planned.items():
        records, counts, emitted = [], Counter(), set()
        for record, rows, multiplicity in plans:
            for repeat in range(multiplicity):
                name = record["file"] if repeat == 0 else f"close-repeat-{repeat}--{record['file']}"
                if name in emitted:
                    raise ValueError("Generated filename collision")
                emitted.add(name)
                dest_image = output / "images" / split / name
                dest_label = output / "labels" / split / (Path(name).stem + ".txt")
                dest_image.parent.mkdir(parents=True, exist_ok=True)
                dest_label.parent.mkdir(parents=True, exist_ok=True)
                dest_image.hardlink_to(source / "images" / split / record["file"])
                shutil.copyfile(source / "labels" / split / (Path(record["file"]).stem + ".txt"),
                                dest_label)
                if (sha256(dest_image) != record["sha256"]
                        or sha256(dest_label) != record["labels_sha256"]):
                    raise ValueError("Source changed during export")
                records.append(dict(record, file=name))
                counts.update(original["classes"][int(r[0])] for r in rows)
        if split == "train":
            result["splits"][split].update(images=records, counts=dict(counts), frames=len(records))
        elif records != original["splits"][split]["images"]:
            raise ValueError("Validation receipt changed")
    dataset = dict(path=str(output), train="images/train", val="images/valid",
                   names=dict(enumerate(original["classes"])))
    (output / "dataset.yaml").write_text(yaml.safe_dump(dataset, sort_keys=False))
    result["dataset_yaml_sha256"] = sha256(output / "dataset.yaml")
    write_json(output / "receipt.json", result)
    return result
