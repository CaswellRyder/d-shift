"""Add explicitly reviewed external training crops; holdout labels/sources unchanged."""
import argparse
from collections import Counter
from pathlib import Path
import shutil

import numpy as np

from dtr.data import read_json, sha256, validate, write_json


def admitted(queue_root, review_path, base_manifest):
    root = Path(queue_root).resolve()
    queue = read_json(root / "review.json")
    review = read_json(review_path)
    if (queue["base_manifest_sha256"] != sha256(base_manifest)
            or review["queue_sha256"] != sha256(root / "review.json")):
        raise ValueError("Queue/review does not match frozen inputs")
    ids = review["admit"]
    if (not ids or len(ids) != len(set(ids)) or set(ids).intersection(review["exclude"])
            or any(type(i) is not int or not 0 <= i < len(queue["samples"]) for i in ids)):
        raise ValueError("Invalid admission")
    base = read_json(base_manifest)
    forbidden = {r["sha256"] for r in base["samples"] if r["split"] != "train"}
    rows = [queue["samples"][i] for i in ids]
    for r in rows:
        path = (root / r["path"]).resolve()
        frame = queue["frames"][r["frame_id"]]
        if (r["split"] != "train" or r["label"] not in base["classes"]
                or Path(r.get("source_image", "")).name.lower().startswith("img_")
                or Path(frame.get("source", "")).name.lower().startswith("img_")
                or not r["session"].startswith("engdes2:indoor-frame-")
                or frame["holdout_hamming"] <= 8 or r["sha256"] in forbidden
                or r["id"] not in frame["crop_ids"]
                or not path.is_relative_to(root) or sha256(path) != r["sha256"]):
            raise ValueError("Invalid external training crop")
    return rows


def checked_hard_negatives(base, base_manifest, root, review):
    from scripts.refine_pi_student import reviewed_negatives
    rows = reviewed_negatives(root,review,sha256(base_manifest))
    allowed = {(r["source_image"],r["session"]) for r in base["samples"] if r["split"] == "train"}
    forbidden = {r["sha256"] for r in base["samples"] if r["split"] != "train" or r["label"] != "background"}
    if any((r["source_image"],r["session"]) not in allowed or r["sha256"] in forbidden for r in rows):
        raise ValueError("Hard negative is outside original training partition")
    return rows


def build(base_manifest, queue_root, review, output, control=False, repetitions=2,
          hard_negative_mode="none", hard_negative_queue=None, hard_negative_review=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if not 1 <= repetitions <= 4:
        raise ValueError("Require 1..4 repetitions")
    config = read_json("configs/balloon-red-blue.json")
    base = validate(base_manifest, config, splits=("train", "val"))
    extra = admitted(queue_root, review, base_manifest)
    if hard_negative_mode not in ("none","control","admit"):
        raise ValueError("Invalid hard-negative mode")
    hard = []
    if hard_negative_mode != "none":
        if not hard_negative_queue or not hard_negative_review:
            raise ValueError("Hard-negative queue and review required")
        hard = checked_hard_negatives(base,base_manifest,hard_negative_queue,hard_negative_review)
    output.mkdir(parents=True)
    samples = []
    for r in base["samples"]:
        relative = f"original/{r['path']}"
        if r["split"] != "test":
            dest = output / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path(base_manifest).parent/r["path"],dest)
        # Test metadata remains for leakage checks; test pixels are not copied/opened.
        samples.append(dict(r,path=relative))
    rng = np.random.default_rng(42)
    train = [r for r in samples if r["split"] == "train"]
    for repetition in range(repetitions):
        for r in extra:
            if control:
                same = [a for a in train if a["label"] == r["label"]]
                chosen = dict(same[int(rng.integers(len(same)))],resample_control=True)
            else:
                relative = f"external/{r['path']}"
                dest = output / relative
                if not dest.exists():
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copyfile(Path(queue_root)/r["path"],dest)
                chosen = dict(r,path=relative,label_origin="AI reviewed source crop",repetition=repetition)
            samples.append(chosen)
    backgrounds = [r for r in samples if r["split"] == "train" and r["label"] == "background"]
    for repetition in range(4):
        for r in hard:
            if hard_negative_mode == "control":
                chosen = dict(backgrounds[int(rng.integers(len(backgrounds)))],hard_negative_control=True)
            else:
                relative = f"hard-negatives/{r['path']}"
                dest = output / relative
                if not dest.exists():
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copyfile(Path(hard_negative_queue)/r["path"],dest)
                chosen = dict(r,path=relative,label_origin="AI reviewed hard negative",repetition=repetition)
            samples.append(chosen)
    doc = dict(base,samples=samples,training_approved=True,deployment_approved=False,
               parent_manifest_sha256=sha256(base_manifest),external_queue_sha256=sha256(Path(queue_root)/"review.json"),
               external_review_sha256=sha256(review),external_control=control,
               hard_negative_mode=hard_negative_mode,
               hard_negative_queue_sha256=sha256(Path(hard_negative_queue)/"review.json") if hard else None,
               hard_negative_review_sha256=sha256(hard_negative_review) if hard else None,
               test_pixels_materialized=False,
               qualification=dict(base["qualification"], external="Two broad training-only filename families; no independently verified sessions",
                                  augmentations="Public source includes flips/noise/blur; not new independent images"))
    write_json(output / "manifest.json",doc)
    validate(output / "manifest.json",config,splits=("train","val"))
    receipt = dict(manifest_sha256=sha256(output/"manifest.json"),script_sha256=sha256(__file__),
                   parent_manifest_sha256=sha256(base_manifest),control=control,repetitions=repetitions,
                   added_entries=len(extra)*repetitions,unique_external_crops=0 if control else len(extra),
                   hard_negative_mode=hard_negative_mode,hard_negative_entries=len(hard)*4,
                   counts=dict(Counter(f"{r['split']}/{r['label']}" for r in samples)),
                   test_evaluated=False,deployment_approved=False)
    write_json(output/"receipt.json",receipt)
    print(receipt)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",required=True)
    p.add_argument("--queue",required=True)
    p.add_argument("--review",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--control",action="store_true")
    p.add_argument("--hard-negative-mode",choices=("none","control","admit"),default="none")
    p.add_argument("--hard-negative-queue")
    p.add_argument("--hard-negative-review")
    args = p.parse_args()
    build(args.manifest,args.queue,args.review,args.output,args.control,
          hard_negative_mode=args.hard_negative_mode,hard_negative_queue=args.hard_negative_queue,
          hard_negative_review=args.hard_negative_review)
