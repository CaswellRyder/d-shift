"""Export AI-reviewed yellow-goal draft COCO, quarantined from training."""

from pathlib import Path

from PIL import Image

from dtr.data import read_json, sha256, write_json


def main():
    root = Path("data/raw/iu-public-inspection")
    out = root / "goal-draft.coco.json"
    if out.exists():
        raise FileExistsError(out)
    review = read_json("configs/iu-goal-seed-review.json")
    labels = ["yellow_circle", "yellow_square", "yellow_triangle"]
    images, annotations = [], []
    for image_id, (filename, boxes) in enumerate(review["images"].items()):
        path = root / filename
        with Image.open(path) as im:
            width, height = im.size
        images.append(
            dict(
                id=image_id,
                file_name=filename,
                width=width,
                height=height,
                sha256=sha256(path),
                session=review["session"],
            )
        )
        for item in boxes:
            x1, y1, x2, y2 = item["box"]
            x, y = x1 * width / 1600, y1 * height / 1600
            w, h = (x2 - x1) * width / 1600, (y2 - y1) * height / 1600
            annotations.append(
                dict(
                    id=len(annotations),
                    image_id=image_id,
                    category_id=labels.index(item["label"]),
                    bbox=[x, y, w, h],
                    area=w * h,
                    iscrowd=0,
                )
            )
    write_json(
        out,
        dict(
            info=review,
            images=images,
            annotations=annotations,
            categories=[dict(id=i, name=name) for i, name in enumerate(labels)],
        ),
    )
    print(
        f"Draft only: {len(images)} real DTR images, {len(annotations)} yellow goals; no training"
    )


if __name__ == "__main__":
    main()
