"""Training-only localization coverage; never labels missing regions as background."""
import argparse
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, write_json
from dtr.tracking import iou
from scripts.audit_public_balloon_export import bounded_read
from scripts.build_indoor_balloon_training import admitted
from scripts.prepare_indoor_balloon_review import ARCHIVE_SHA
from scripts.research_balloon_search import experimental


def run(root, review, manifest, archive_path):
    root = Path(root)
    rows = [r for r in admitted(root, review, manifest) if r["label"] != "background"]
    if sha256(archive_path) != ARCHIVE_SHA:
        raise ValueError("Archive changed")
    with zipfile.ZipFile(archive_path) as archive:
        doc = json.loads(bounded_read(archive, "train/_annotations.coco.json", 50_000_000))
    annotations = {r["id"]: r for r in doc["annotations"]}
    sources = {r["id"]: r for r in doc["images"]}
    variants = {v: [] for v in ("baseline", "mser", "mser_chromatic", "mser_blue_red")}
    for frame in read_json(root / "review.json")["frames"]:
        targets = [r for r in rows if r["frame_id"] == frame["id"]]
        if not targets:
            continue
        path = root / frame["path"]
        if sha256(path) != frame["sha256"] or Path(frame["source"]).name.lower().startswith("img_"):
            raise ValueError("Frame changed or reserved IMG family")
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"))
        for variant, results in variants.items():
            found = experimental(rgb, variant)
            for row in targets:
                annotation = annotations[row["annotation_id"]]
                source = sources[annotation["image_id"]]
                if "train/" + source["file_name"] != row["source_image"]:
                    raise ValueError("Annotation identity changed")
                x, y, w, h = annotation["bbox"]
                box = [x*320/source["width"], y*240/source["height"],
                       (x+w)*320/source["width"], (y+h)*240/source["height"]]
                group = 0 if row["label"] == "red_balloon" else 1
                overlap = max((iou(box, c["box"]) for c in found if c["color_group"] == group), default=0.)
                results.append(dict(source=row["source_image"], annotation_id=row["annotation_id"],
                                    label=row["label"], best_iou=overlap, covered=overlap >= .5))
    return dict(scope="Admitted training-positive localization only; no false-positive or accuracy claim",
                queue_sha256=sha256(root/"review.json"), review_sha256=sha256(review),
                manifest_sha256=sha256(manifest), script_sha256=sha256(__file__),
                search_sha256=sha256("scripts/research_balloon_search.py"), deployment_approved=False,
                variants={v: dict(rows=rs, counts={label: dict(total=sum(r["label"] == label for r in rs),
                    covered=sum(r["label"] == label and r["covered"] for r in rs))
                    for label in ("red_balloon", "blue_balloon")}) for v, rs in variants.items()})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("queue", "review", "manifest", "archive", "output"):
        p.add_argument(f"--{name}", required=True)
    args = p.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    result = run(args.queue, args.review, args.manifest, args.archive)
    write_json(args.output, result)
    print({v: r["counts"] for v, r in result["variants"].items()})
