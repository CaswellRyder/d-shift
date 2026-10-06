"""Inspect missed TRAINING goals and separate candidate-budget loss from localization loss."""

from collections import defaultdict, Counter
import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from dtr.data import read_json, write_json, sha256
from dtr.tracking import iou
from dtr.vision import PROPOSAL_PROFILES, proposals, color_mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROPOSAL_PROFILES, default="v2")
    parser.add_argument("--output", default="data/proposal-audit-20261005")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--task", choices=["goal", "balloon"], default="goal")
    parser.add_argument("--goal-colors", choices=["orange", "all"], default="orange")
    parser.add_argument("--width", type=int, choices=[320, 640], default=320,
                        help="Train-only geometry experiment; does not change viewer resolution")
    args = parser.parse_args()
    width, height = args.width, args.width * 3 // 4
    root = Path("data/roboflow-dtr-v10-grouped/coco/train")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir()
    shutil.copyfile("src/dtr/vision.py", output / "vision.py")
    shutil.copyfile(__file__, output / "audit.py")
    doc = read_json(root / "_annotations.coco.json")
    names = {c["id"]: c["name"] for c in doc["categories"]}
    anns = defaultdict(list)
    for ann in doc["annotations"]:
        anns[ann["image_id"]].append(ann)
    counts, sizes, missed = defaultdict(Counter), defaultdict(Counter), []
    images = sorted(doc["images"], key=lambda im: im["file_name"])[::8]
    for info in images:
        with Image.open(root / info["file_name"]) as im:
            rgb = cv2.resize(np.asarray(im.convert("RGB")), (width, height))
        limited, expanded = (
            proposals(rgb, args.task, limit=args.limit, profile=args.profile),
            proposals(rgb, args.task, limit=64, profile=args.profile),
        )
        for ann in anns[info["id"]]:
            name = names[ann["category_id"]]
            relevant = (
                "Orange" in name or (args.goal_colors == "all" and "Yellow" in name)
                if args.task == "goal" else name.endswith("Balloon")
            )
            if not relevant:
                continue
            x, y, w, h = ann["bbox"]
            box = [
                x * width / info["width"],
                y * height / info["height"],
                (x + w) * width / info["width"],
                (y + h) * height / info["height"],
            ]
            best12 = max((iou(box, c["box"]) for c in limited), default=0)
            best64 = max((iou(box, c["box"]) for c in expanded), default=0)
            counts[name]["targets"] += 1
            counts[name][f"covered{args.limit}"] += best12 >= 0.5
            counts[name]["covered64"] += best64 >= 0.5
            longest = max(box[2] - box[0], box[3] - box[1])
            size_bin = "under16px" if longest < 16 else "16to31px" if longest < 32 else "32pluspx"
            size_key = f"{name}/{size_bin}"
            sizes[size_key]["targets"] += 1
            sizes[size_key][f"covered{args.limit}"] += best12 >= 0.5
            sizes[size_key]["covered64"] += best64 >= 0.5
            if best12 < 0.5:
                missed.append(
                    dict(
                        file=info["file_name"],
                        label=name,
                        box=box,
                        best_budget=best12,
                        best64=best64,
                    )
                )
    rng = np.random.default_rng(20261005)
    selected = [
        missed[int(i)] for i in rng.choice(len(missed), min(48, len(missed)), replace=False)
    ]
    for page in range((len(selected) + 15) // 16):
        sheet = Image.new("RGB", (4 * 280, 4 * 160), "white")
        draw = ImageDraw.Draw(sheet)
        for n, row in enumerate(selected[page * 16 : (page + 1) * 16]):
            with Image.open(root / row["file"]) as im:
                rgb = cv2.resize(np.asarray(im.convert("RGB")), (width, height))
            x1, y1, x2, y2 = row["box"]
            pad = max(5, int(max(x2 - x1, y2 - y1) * 0.25))
            a, b, c, d = (
                max(0, int(x1) - pad),
                max(0, int(y1) - pad),
                min(width, int(x2) + pad),
                min(height, int(y2) + pad),
            )
            if c <= a or d <= b:
                continue
            crop = Image.fromarray(rgb[b:d, a:c])
            ImageDraw.Draw(crop).rectangle([x1 - a, y1 - b, x2 - a, y2 - b], outline="cyan")
            mask = Image.fromarray(color_mask(rgb, args.task, args.profile)[b:d, a:c])
            x, y = n % 4 * 280, n // 4 * 160
            sheet.paste(ImageOps.contain(crop, (136, 125)), (x, y))
            sheet.paste(ImageOps.contain(mask, (136, 125)), (x + 140, y))
            draw.text((x, y + 128), f"{page * 16 + n} {row['label']}", fill="black")
            draw.text(
                (x, y + 141),
                f"IoU{args.limit}={row['best_budget']:.2f} 64={row['best64']:.2f}",
                fill="black",
            )
        sheet.save(output / f"missed-{page}.jpg")
    write_json(
        output / "report.json",
        dict(
            split="train",
            profile=args.profile,
            task=args.task,
            goal_colors=args.goal_colors,
            proposal_limit=args.limit,
            image_size=[width, height],
            iou_threshold=0.5,
            frames=[im["file_name"] for im in images],
            scope="Train-only box coverage; no teacher inference or runtime qualification",
            deployment_approved=False,
            test_evaluated=False,
            annotation_sha256=sha256(root / "_annotations.coco.json"),
            vision_sha256=sha256("src/dtr/vision.py"),
            counts=counts,
            size_counts=sizes,
            review=selected,
        ),
    )
    print(dict(counts))


if __name__ == "__main__":
    main()
