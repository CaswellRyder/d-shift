"""Visualize cached validation errors; diagnostics only, never new training labels."""

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from dtr.data import read_json, write_json, sha256
from dtr.tracking import iou


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    report = read_json(args.report)
    if report["split"] != "val" or report["test_evaluated"]:
        raise ValueError("Only development validation receipts supported")
    source = Path("data/roboflow-dtr-v10-grouped/coco/valid")
    labels = source / "_annotations.coco.json"
    if sha256(labels) != report["annotation_sha256"]:
        raise ValueError("Annotation checksum differs from evaluation")
    doc = read_json(labels)
    names = {c["id"]: c["name"] for c in doc["categories"]}
    annotations = defaultdict(list)
    for a in doc["annotations"]:
        annotations[a["image_id"]].append(a)
    images = {im["file_name"]: im for im in doc["images"]}
    groups = defaultdict(list)
    for task in ("balloon", "goal"):
        aliases = read_json(f"configs/{task}.json")["aliases"]
        for frame in report["tasks"][task]["frames"]:
            info = images[frame["file"]]
            truth = []
            for ann in annotations[info["id"]]:
                label = aliases.get(names[ann["category_id"]])
                if label:
                    x, y, w, h = ann["bbox"]
                    truth.append(
                        dict(
                            label=label,
                            box=[
                                x * 320 / info["width"],
                                y * 240 / info["height"],
                                (x + w) * 320 / info["width"],
                                (y + h) * 240 / info["height"],
                            ],
                        )
                    )
            for pred in frame["accepted"]:
                if not pred["matched"]:
                    groups[f"{pred['label']}-unmatched"].append(
                        dict(
                            file=frame["file"],
                            task=task,
                            kind="unmatched prediction",
                            **pred,
                            truth=truth,
                        )
                    )
            used = set()
            for pred in sorted(frame["accepted"], key=lambda p: -p["score"]):
                match = max(
                    (
                        i
                        for i, t in enumerate(truth)
                        if i not in used and t["label"] == pred["label"]
                    ),
                    key=lambda i: iou(truth[i]["box"], pred["box"]),
                    default=None,
                )
                if match is not None and iou(truth[match]["box"], pred["box"]) >= 0.5:
                    used.add(match)
            for i, actual in enumerate(truth):
                if i not in used:
                    groups[f"{actual['label']}-missed"].append(
                        dict(
                            file=frame["file"],
                            task=task,
                            kind="unmatched annotation",
                            score=None,
                            **actual,
                            truth=truth,
                        )
                    )
    output.mkdir(parents=True)
    receipt = dict(
        split="val",
        training_approved=False,
        test_evaluated=False,
        scope="Error triage, not corrected labels. Upstream omissions/conflicts may explain errors.",
        report_sha256=sha256(args.report),
        groups={},
    )
    for key, rows in sorted(groups.items()):
        # Reproducible sample, one crop per frame; confidence ranking is not label evidence.
        chosen, seen = [], set()
        order = (
            sorted(rows, key=lambda r: -(r["score"] or 0)) if key.endswith("unmatched") else rows
        )
        for row in order:
            if row["file"] not in seen:
                chosen.append(row)
                seen.add(row["file"])
            if len(chosen) == 16:
                break
        sheet = Image.new("RGB", (1120, 640), "white")
        draw = ImageDraw.Draw(sheet)
        for index, row in enumerate(chosen):
            with Image.open(source / row["file"]) as im:
                rgb = cv2.resize(np.asarray(im.convert("RGB")), (320, 240))
            marked = Image.fromarray(rgb)
            pen = ImageDraw.Draw(marked)
            for actual in row["truth"]:
                pen.rectangle(actual["box"], outline="cyan", width=1)
            pen.rectangle(row["box"], outline="red", width=2)
            x1, y1, x2, y2 = row["box"]
            box = [
                max(0, int(x1) - 5),
                max(0, int(y1) - 5),
                min(320, int(x2) + 5),
                min(240, int(y2) + 5),
            ]
            crop = Image.fromarray(rgb).crop(box)
            x, y = index % 4 * 280, index // 4 * 160
            sheet.paste(ImageOps.contain(marked, (136, 120)), (x, y))
            sheet.paste(ImageOps.contain(crop, (136, 120)), (x + 140, y))
            draw.text((x, y + 123), f"{index}: {row['label']}", fill="black")
            draw.text(
                (x, y + 137),
                f"score {row['score']:.3f}"
                if row["score"] is not None
                else "unmatched upstream label",
                fill="black",
            )
        sheet.save(output / f"{key}.jpg")
        write_json(output / f"{key}.json", chosen)
        receipt["groups"][key] = dict(total=len(rows), shown=len(chosen))
    write_json(output / "review.json", receipt)
    print(receipt)


if __name__ == "__main__":
    main()
