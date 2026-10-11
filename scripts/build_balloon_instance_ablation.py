"""Matched, reviewed instance-crop additions; identical class counts for control."""
import argparse
from collections import Counter
from pathlib import Path
import shutil

import numpy as np

from dtr.data import read_json, sha256, validate, write_json
from scripts.prepare_balloon_instance_review import checked_rows
from scripts.audit_balloon_search_stages import training_source


def build(manifest, queue, review, output, mode, repetitions=4):
    if mode not in ("admit", "control") or type(repetitions) is not int or not 1 <= repetitions <= 4:
        raise ValueError("Require admit/control and 1..4 repetitions")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    config = read_json("configs/balloon-red-blue.json")
    base = validate(manifest, config, splits=("train", "val"))
    if base.get("training_approved") is not True or base.get("synthetic", True):
        raise ValueError("Require approved real base")
    extra = checked_rows(manifest, queue, review)
    train = [r for r in base["samples"] if r["split"] == "train"]
    for row in train:
        training_source(row["source_image"])
        if "instance_ablation_mode" in row:
            raise ValueError("Instance crops already added")
    known_labels = {}
    for row in train:
        known_labels.setdefault(row["sha256"], set()).add(row["label"])
    if any(r["sha256"] in known_labels and known_labels[r["sha256"]] != {r["label"]} for r in extra):
        raise ValueError("Reviewed crop conflicts with existing training label")
    output.mkdir(parents=True)

    def copy(row, root, prefix):
        relative = f"{prefix}/{row['path']}"
        if row["split"] != "test":
            target = output / relative
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(Path(root) / row["path"], target)
        return dict(row, path=relative)

    samples = [copy(r, Path(manifest).parent, "base") for r in base["samples"]]
    pools = {label: [r for r in samples if r["split"] == "train" and r["label"] == label]
             for label in base["classes"]}
    rng = np.random.default_rng(47)
    for repetition in range(repetitions):
        for row in extra:
            if mode == "admit":
                selected = copy(row, queue, "instance")
            else:
                choices = pools[row["label"]]
                selected = dict(choices[int(rng.integers(len(choices)))])
            samples.append(dict(selected, instance_ablation_mode=mode, instance_repetition=repetition))
    doc = dict(base, samples=samples, parent_manifest_sha256=sha256(manifest),
               instance_queue_sha256=sha256(Path(queue)/"review.json"), instance_review_sha256=sha256(review),
               instance_mode=mode, instance_repetitions=repetitions,
               test_pixels_materialized=False, deployment_approved=False)
    write_json(output / "manifest.json", doc)
    validate(output / "manifest.json", config, splits=("train", "val"))
    receipt = dict(mode=mode, manifest_sha256=sha256(output/"manifest.json"),
                   parent_manifest_sha256=sha256(manifest), queue_sha256=sha256(Path(queue)/"review.json"),
                   review_sha256=sha256(review), script_sha256=sha256(__file__),
                   guard_sha256=sha256("scripts/prepare_balloon_instance_review.py"),
                   entries_added=len(extra)*repetitions, unique_added_crops=len(extra) if mode == "admit" else 0,
                   counts=dict(Counter(f"{r['split']}/{r['label']}" for r in samples)),
                   test_evaluated=False, deployment_approved=False)
    write_json(output / "receipt.json", receipt)
    return receipt


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "queue", "review", "output"):
        p.add_argument(f"--{name}", required=True)
    p.add_argument("--mode", choices=("admit", "control"), required=True)
    p.add_argument("--repetitions", type=int, default=4)
    a = p.parse_args()
    print(build(a.manifest, a.queue, a.review, a.output, a.mode, a.repetitions))
