"""Copy the admitted frames of a reviewed development panel into a sweep bundle.

The output is a `research_pixel_photometric.py --base` directory. Only frames the
frozen review admits are copied, with their provisional upstream boxes as truth.
"""
import argparse
from pathlib import Path
import shutil

from dtr.data import read_json, sha256, write_json


def bundle(queue_root, review_path, output, split="development_panel2"):
    queue_root, output = Path(queue_root), Path(output)
    review = read_json(review_path)
    if review.get("evaluation_approved") is not True or review.get("training_approved") is not False:
        raise ValueError("Review is not an evaluation-only approval")
    if review["queue_sha256"] != sha256(queue_root / "review.json"):
        raise ValueError("Queue changed after review")
    if set(review["admit"]) & set(review["exclude"]):
        raise ValueError("Frame both admitted and excluded")
    if output.exists():
        raise FileExistsError(output)
    queue = read_json(queue_root / "review.json")
    (output / "frames").mkdir(parents=True)
    frames = []
    for i in review["admit"]:
        row = queue["frames"][i]
        if sha256(queue_root / row["path"]) != row["sha256"]:
            raise ValueError("Panel frame changed")
        path = f"frames/{i:03d}.png"
        shutil.copy(queue_root / row["path"], output / path)
        frames.append(dict(path=path, source=row["source"], panel="indoor",
                           truth=row["provisional_truth"], queue_id=i))
    write_json(output / "inputs.json", dict(split=split, frames=frames, models={},
                                            queue_sha256=review["queue_sha256"],
                                            review_sha256=sha256(review_path)))
    return frames


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--queue", required=True)
    p.add_argument("--review", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--split", default="development_panel2")
    args = p.parse_args()
    frames = bundle(args.queue, args.review, args.output, args.split)
    print(dict(frames=len(frames), targets=sum(len(f["truth"]) for f in frames)))
