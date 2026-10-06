"""Build reviewed train-only adaptation data, binding manual decisions to queue hashes."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, validate, write_json


def select_reviewed(queue, decision):
    indices = decision["admit"]
    if len(set(indices)) != len(indices) or any(
        not isinstance(i, int) or not 0 <= i < len(queue) for i in indices
    ):
        raise ValueError("Invalid or repeated review indices")
    rows = [queue[i] for i in indices]
    if any(r["label"] != "background" or r["split"] != "train" for r in rows):
        raise ValueError("Review admits only training negatives")
    return rows


def build(task, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    base = Path("data/roboflow-dtr-v10-grouped") / task
    raw = Path("data/dtr-proposals-20261005") / task
    review_path = Path("configs/proposal-negative-review-20261005.json")
    review = read_json(review_path)
    decision = review[task]
    if sha256(raw / "negative-review-queue.json") != decision["queue_sha256"]:
        raise ValueError("Review queue changed since visual inspection")
    original = read_json(base / "manifest.json")
    receipt = read_json(raw / "mining-receipt.json")
    if sha256(base / "manifest.json") != receipt["base_manifest_sha256"]:
        raise ValueError("Base manifest changed since mining")
    mined = read_json(raw / "mined.json")
    queue = read_json(raw / "negative-review-queue.json")
    negatives = select_reviewed(queue, decision)
    if any(r not in mined for r in negatives):
        raise ValueError("Review entries differ from mined candidates")
    excluded = set(decision["exclude_training_sources"])
    retained = [
        r
        for r in original["samples"]
        if not (r["split"] == "train" and r.get("source_image") in excluded)
    ]
    positives = [
        r for r in mined if r["label"] != "background" and r["source_image"] not in excluded
    ]
    negatives = [r for r in negatives if r["source_image"] not in excluded]
    if any(r["split"] != "train" for r in positives):
        raise ValueError("New positives must be train-only")
    output.mkdir(parents=True)
    for source, rows in [(base, retained), (raw, positives + negatives)]:
        for row in rows:
            dest = output / row["path"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                dest.hardlink_to(source / row["path"])
    train = [r for r in retained if r["split"] == "train"] + positives
    train += negatives * review["negative_repetitions"]
    np.random.default_rng(42).shuffle(train)
    held_out = [r for r in retained if r["split"] != "train"]
    rows = train + held_out
    for split in ("val", "test"):
        assert [r for r in rows if r["split"] == split] == [
            r for r in original["samples"] if r["split"] == split
        ]
    evidence = dict(
        reviewer=review["reviewer"],
        policy=review["policy"],
        review_config_sha256=sha256(review_path),
        queue_sha256=decision["queue_sha256"],
        base_manifest_sha256=sha256(base / "manifest.json"),
        mined_sha256=sha256(raw / "mined.json"),
        builder_sha256=sha256(__file__),
        reviewed_negative_candidates=len(queue),
        unique_negatives_admitted=len(negatives),
        negative_repetitions=review["negative_repetitions"],
        unique_positives_admitted=dict(Counter(r["label"] for r in positives)),
        original_training_crops_excluded=len(original["samples"]) - len(retained),
        excluded_training_sources=sorted(excluded),
        val_and_test_samples_unchanged=True,
        test_evaluated=False,
    )
    write_json(output / "review-receipt.json", evidence)
    write_json(
        output / "manifest.json",
        {
            **original,
            "training_approved": True,
            "samples": rows,
            "qualification": {**original["qualification"], "proposal_review": evidence},
        },
    )
    validate(output / "manifest.json", read_json(f"configs/{task}.json"))
    # Existing raw mining outputs predate the automatic quarantine gate.
    raw_doc = read_json(raw / "manifest.json")
    raw_doc["training_approved"] = False
    write_json(raw / "manifest.json", raw_doc)
    print(task, evidence, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=["balloon", "goal"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    build(args.task, args.output)
