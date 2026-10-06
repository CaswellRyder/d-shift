"""Create review-only crop sheets from the public Matterport balloon release.

Preserves upstream polygon annotations; does not infer color labels or create splits.
"""

import io
import json
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw


def main():
    root = Path("data/raw/matterport-balloon")
    output = root / "review"
    output.mkdir(exist_ok=False)
    rows = []
    with zipfile.ZipFile(root / "balloon_dataset.zip") as archive:
        for split in ("train", "val"):
            records = json.loads(archive.read(f"balloon/{split}/via_region_data.json"))
            for record in records.values():
                source = f"balloon/{split}/{record['filename']}"
                with Image.open(io.BytesIO(archive.read(source))) as opened:
                    image = opened.convert("RGB")
                regions = record["regions"]
                regions = regions.values() if isinstance(regions, dict) else regions
                for region in regions:
                    shape = region["shape_attributes"]
                    xs, ys = shape["all_points_x"], shape["all_points_y"]
                    box = (min(xs), min(ys), max(xs) + 1, max(ys) + 1)
                    index = len(rows)
                    image.crop(box).save(output / f"crop-{index:03d}.png")
                    rows.append(
                        {
                            "id": index,
                            "source": source,
                            "box_xyxy": box,
                            "polygon": list(zip(xs, ys)),
                            "upstream_label": "balloon",
                        }
                    )
    (output / "index.json").write_text(json.dumps(rows, indent=2) + "\n")
    for start in range(0, len(rows), 80):
        sheet = Image.new("RGB", (1000, 10 * 115), "#dddddd")
        for j, row in enumerate(rows[start : start + 80]):
            crop = Image.open(output / f"crop-{row['id']:03d}.png")
            crop.thumbnail((120, 90))
            x, y = (j % 8) * 125, (j // 8) * 115
            sheet.paste(crop, (x, y + 20))
            ImageDraw.Draw(sheet).text((x + 2, y + 2), str(row["id"]), fill="black")
        sheet.save(output / f"sheet-{start:03d}.jpg")
    print({"images": 74, "annotated_balloon_crops": len(rows), "review": str(output)})


if __name__ == "__main__":
    main()
