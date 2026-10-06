"""Create bounded full visual-review queues; only reviewed negatives can be admitted."""

import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from dtr.data import read_json, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True, choices=["balloon", "goal"])
    args = p.parse_args()
    root = Path("data/dtr-proposals-20261005") / args.task
    negatives = [r for r in read_json(root / "mined.json") if r["label"] == "background"]
    rng = np.random.default_rng(20261005)
    chosen = [
        negatives[int(i)]
        for i in rng.choice(len(negatives), min(256, len(negatives)), replace=False)
    ]
    write_json(root / "negative-review-queue.json", chosen)
    for page in range((len(chosen) + 63) // 64):
        sheet = Image.new("RGB", (8 * 140, 8 * 155), "white")
        draw = ImageDraw.Draw(sheet)
        for n, row in enumerate(chosen[page * 64 : (page + 1) * 64]):
            with Image.open(root / row["path"]) as im:
                tile = ImageOps.contain(im, (136, 132))
            x, y = n % 8 * 140, n // 8 * 155
            sheet.paste(tile, (x, y))
            draw.text((x, y + 134), str(page * 64 + n), fill="black")
        sheet.save(root / f"negative-review-{page}.jpg")
    print(args.task, len(chosen))


if __name__ == "__main__":
    main()
