"""Freeze color-preserving TRAIN views so teacher and student share exact geometry.

No synthetic backgrounds, new labels, test pixels, inference or device access.
Distillation must compute fresh teacher logits for this manifest; a parent cache
is intentionally incompatible. The control repeats each identical source entry.
"""
import argparse
from collections import Counter
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, validate, write_json


def transform(rgb, code):
    """D4 symmetry on the entire rectangular crop; no truncation or hue changes."""
    if (type(code) is not int or not 0 <= code < 8 or rgb.ndim != 3
            or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1 or rgb.dtype != np.uint8):
        raise ValueError("Require uint8 RGB crop and integer symmetry 0..7")
    result = np.rot90(rgb, code % 4)
    if code >= 4:
        result = result[:, ::-1]
    return np.ascontiguousarray(result)


def checked_base(manifest, config):
    doc = validate(manifest, config, splits=("train", "val"))
    if (doc.get("synthetic", True) or doc.get("training_approved") is not True
            or config["task"] != "balloon" or doc.get("paired_augmentation") is not None):
        raise ValueError("Require approved real balloon base without prior paired augmentation")
    heldout = [r for r in doc["samples"] if r["split"] != "train"]
    for row in (r for r in doc["samples"] if r["split"] == "train"):
        source = row.get("source_image", "")
        if not source or Path(source).name.lower().startswith("img_"):
            raise ValueError("Missing training source or reserved IMG family")
        for key in ("source_image", "source_sha256", "source_group"):
            if row.get(key) and any(row[key] == h.get(key) for h in heldout):
                raise ValueError("Training source conflicts with holdout")
    labels = {}
    for row in doc["samples"]:
        old = labels.setdefault(row["sha256"], row["label"])
        if old != row["label"]:
            raise ValueError("Conflicting labels for identical crop")
    return doc


def build(manifest, output, control=False, seed=42):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("Require integer augmentation seed 0..2**32-1")
    config = read_json("configs/balloon-red-blue.json")
    doc = checked_base(manifest, config)
    root = Path(manifest).resolve().parent
    parent_hash = sha256(manifest)
    output.mkdir(parents=True)
    samples = []
    for row in doc["samples"]:
        relative = f"base/{row['path']}"
        if row["split"] != "test":
            dest = output / relative
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(root / row["path"], dest)
        samples.append(dict(row, path=relative))

    rng = np.random.default_rng(seed)
    train = [(i, r) for i, r in enumerate(doc["samples"]) if r["split"] == "train"]
    for i, row in train:
        # One non-identity geometric view per exposure in both seeded arms.
        proposed_code = int(rng.integers(1, 8))
        code = 0 if control else proposed_code
        if control:
            relative, digest = f"base/{row['path']}", row["sha256"]
        else:
            relative = f"paired/{i:05d}.png"
            dest = output / relative
            dest.parent.mkdir(exist_ok=True)
            with Image.open(output / "base" / row["path"]) as im:
                rgb = np.asarray(im.convert("RGB"))
            Image.fromarray(transform(rgb, code)).save(dest)
            digest = sha256(dest)
        samples.append(dict(row, path=relative, sha256=digest,
                            label_origin="Reviewed parent label; whole-crop pixel permutation",
                            augmentation_parent_row=i, augmentation_parent_sha256=row["sha256"],
                            augmentation_code=code, proposed_augmentation_code=proposed_code))
    if sha256(manifest) != parent_hash:
        raise ValueError("Training base changed while building")
    qualification = dict(doc.get("qualification", {}),
        augmentations="Fixed TRAIN-only whole-crop right-angle rotation/reflection; not independent data",
        negatives="Inherited reviewed backgrounds and non-balloon clutter; see parent and sample provenance")
    result = dict(doc, samples=samples, parent_manifest_sha256=parent_hash,
                  paired_augmentation=dict(kind="identity-control" if control else "fixed-D4",
                      seed=seed, extra_views_per_training_entry=1,
                      targets="Recompute teacher logits from these exact files; never reuse parent cache"),
                  qualification=qualification, test_pixels_materialized=False, training_approved=False,
                  deployment_approved=False)
    labels = {}
    for row in samples:
        if labels.setdefault(row["sha256"], row["label"]) != row["label"]:
            raise ValueError("Augmentation introduces conflicting crop labels")
    write_json(output / "manifest.json", result)
    validate(output / "manifest.json", config, splits=("train", "val"))
    result["training_approved"] = True
    write_json(output / "manifest.json", result)
    receipt = dict(manifest_sha256=sha256(output / "manifest.json"),
                   parent_manifest_sha256=parent_hash, builder_sha256=sha256(__file__),
                   control=control, seed=seed, added_entries=len(train),
                   unique_training_crops=len({r["sha256"] for r in samples if r["split"] == "train"}),
                   counts=dict(Counter(f"{r['split']}/{r['label']}" for r in samples)),
                   test_evaluated=False, deployment_approved=False)
    write_json(output / "receipt.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--control", action="store_true")
    args = parser.parse_args()
    print(build(args.manifest, args.output, args.control, args.seed))
