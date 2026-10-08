"""Append a bounded dose of reviewed TRAIN clutter or matched background controls."""
import argparse
from collections import Counter
from pathlib import Path
import shutil

import numpy as np

from dtr.data import read_json, sha256, validate, write_json
from scripts.build_indoor_balloon_training import checked_hard_negatives


def build(manifest, original_manifest, queue, review, output, mode, repetitions=1):
    if mode not in ("admit", "control") or type(repetitions) is not int or not 1 <= repetitions <= 4:
        raise ValueError("Require admit/control and 1..4 repetitions")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    config = read_json("configs/balloon-red-blue.json")
    base = validate(manifest, config, splits=("train", "val"))
    original = validate(original_manifest, config, splits=("train", "val"))
    if (base.get("synthetic", True) or base.get("training_approved") is not True
            or base.get("parent_manifest_sha256") != sha256(original_manifest)
            or base.get("hard_negative_mode") != "none"):
        raise ValueError("Require approved expanded data without previously added clutter")
    negatives = checked_hard_negatives(original, original_manifest, queue, review)
    for row in [r for r in base["samples"] if r["split"] == "train"] + negatives:
        if Path(row.get("source_image", "")).name.lower().startswith("img_"):
            raise ValueError("Reserved IMG family cannot enter training")
    forbidden = {r["sha256"] for r in base["samples"]
                 if r["split"] != "train" or r["label"] != "background"}
    heldout_sessions = {r["session"] for r in base["samples"] if r["split"] != "train"}
    if any(r["sha256"] in forbidden or r["session"] in heldout_sessions for r in negatives):
        raise ValueError("Clutter conflicts with expanded positives or holdout")
    output.mkdir(parents=True)
    samples = []

    def copy(row, root, prefix):
        relative = f"{prefix}/{row['path']}"
        if row["split"] != "test":
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copyfile(Path(root) / row["path"], target)
        return dict(row, path=relative)

    for row in base["samples"]:
        samples.append(copy(row, Path(manifest).parent, "base"))
    backgrounds = [r for r in samples if r["split"] == "train" and r["label"] == "background"]
    rng = np.random.default_rng(42)
    for repetition in range(repetitions):
        for row in negatives:
            if mode == "admit":
                added = copy(row, queue, "clutter")
            else:
                added = dict(backgrounds[int(rng.integers(len(backgrounds)))])
            samples.append(dict(added, clutter_ablation_mode=mode, clutter_repetition=repetition))
    doc = dict(base, samples=samples, parent_manifest_sha256=sha256(manifest),
               original_manifest_sha256=sha256(original_manifest), hard_negative_mode=mode,
               hard_negative_repetitions=repetitions, hard_negative_entries=len(negatives)*repetitions,
               hard_negative_queue_sha256=sha256(Path(queue) / "review.json"),
               hard_negative_review_sha256=sha256(review), test_pixels_materialized=False,
               deployment_approved=False)
    write_json(output / "manifest.json", doc)
    validate(output / "manifest.json", config, splits=("train", "val"))
    receipt = dict(manifest_sha256=sha256(output / "manifest.json"),
                   parent_manifest_sha256=sha256(manifest), original_manifest_sha256=sha256(original_manifest),
                   builder_sha256=sha256(__file__),
                   review_guard_sha256=sha256("scripts/build_indoor_balloon_training.py"),
                   mode=mode, repetitions=repetitions, added_entries=len(negatives)*repetitions,
                   counts=dict(Counter(f"{r['split']}/{r['label']}" for r in samples)),
                   test_evaluated=False, deployment_approved=False)
    write_json(output / "receipt.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "original-manifest", "queue", "review", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--mode", choices=("admit", "control"), required=True)
    parser.add_argument("--repetitions", type=int, default=1)
    args = parser.parse_args()
    print(build(args.manifest, args.original_manifest, args.queue, args.review,
                args.output, args.mode, args.repetitions))
