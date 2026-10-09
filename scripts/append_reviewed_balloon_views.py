"""Append a reviewed expansion queue to a frozen training manifest, or matched controls."""
import argparse
from collections import Counter
from pathlib import Path
import shutil

import numpy as np

from dtr.data import read_json, sha256, validate, write_json
from scripts.build_indoor_balloon_training import admitted


def build(manifest, original_manifest, queue, review, output, control=False):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    config = read_json("configs/balloon-red-blue.json")
    base = validate(manifest, config, splits=("train", "val"))
    if (base.get("synthetic", True) or base.get("training_approved") is not True
            or base.get("parent_manifest_sha256") != sha256(original_manifest)):
        raise ValueError("Require approved expanded manifest with original parent identity")
    decision = read_json(review)
    queue_doc = read_json(Path(queue) / "review.json")
    decided = decision["admit"] + decision["exclude"]
    if (decision.get("training_approved") is not True
            or any(type(i) is not int for i in decided)
            or len(set(decided)) != len(decided)
            or set(decided) != set(range(len(queue_doc["samples"])))):
        raise ValueError("Require explicit complete training-only review")
    extra = admitted(queue, review, original_manifest)
    old_groups = {r.get("source_group") for r in base["samples"] if r["split"] == "train"}
    heldout_sources = {r.get("source_image") for r in base["samples"] if r["split"] != "train"}
    for row in extra:
        if row.get("source_group") in old_groups or row.get("source_image") in heldout_sources:
            raise ValueError("Expansion repeats a prior group or held-out source")
        if any(row["sha256"] == r["sha256"] and (r["split"] != "train" or r["label"] != row["label"])
               for r in base["samples"]):
            raise ValueError("Expansion conflicts with heldout or existing label")
    if any(Path(r.get("source_image", "")).name.lower().startswith("img_")
           for r in base["samples"] if r["split"] == "train"):
        raise ValueError("Reserved IMG family in training base")
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
    train = [r for r in samples if r["split"] == "train"]
    rng = np.random.default_rng(42)
    for row in extra:
        if control:
            same = [r for r in train if r["label"] == row["label"]]
            chosen = dict(same[int(rng.integers(len(same)))], expansion_control=True)
        else:
            chosen = copy(row, queue, "expansion")
        samples.append(chosen)
    doc = dict(base, samples=samples, parent_manifest_sha256=sha256(manifest),
               original_manifest_sha256=sha256(original_manifest),
               expansion_queue_sha256=sha256(Path(queue) / "review.json"),
               expansion_review_sha256=sha256(review), expansion_control=control,
               test_pixels_materialized=False, deployment_approved=False)
    write_json(output / "manifest.json", doc)
    validate(output / "manifest.json", config, splits=("train", "val"))
    receipt = dict(manifest_sha256=sha256(output / "manifest.json"),
                   parent_manifest_sha256=sha256(manifest), builder_sha256=sha256(__file__),
                   admission_guard_sha256=sha256("scripts/build_indoor_balloon_training.py"),
                   control=control, added_entries=len(extra),
                   counts=dict(Counter(f"{r['split']}/{r['label']}" for r in samples)),
                   test_evaluated=False, deployment_approved=False)
    write_json(output / "receipt.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "original-manifest", "queue", "review", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--control", action="store_true")
    args = parser.parse_args()
    print(build(args.manifest, args.original_manifest, args.queue, args.review, args.output, args.control))
