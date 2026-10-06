"""Bounded train-only mask diagnostics. Does not read the held-out test images."""

from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
from dtr.data import read_json

root = Path("data/roboflow-dtr-v10-grouped/coco/train")
doc = read_json(root / "_annotations.coco.json")
categories = {a["id"]: a["name"] for a in doc["categories"]}
anns = defaultdict(list)
for ann in doc["annotations"]:
    anns[ann["image_id"]].append(ann)
images = [
    im
    for im in doc["images"]
    if any(categories[a["category_id"]] == "Orange Square Goal" for a in anns[im["id"]])
][::40][:6]
sheet = Image.new("RGB", (4 * 320, len(images) * 260), "white")
draw = ImageDraw.Draw(sheet)
for row, im in enumerate(images):
    rgb = cv2.resize(np.asarray(Image.open(root / im["file_name"]).convert("RGB")), (320, 240))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    annotated = rgb.copy()
    for a in anns[im["id"]]:
        if "Orange" in categories[a["category_id"]]:
            x, y, w, h = a["bbox"]
            cv2.rectangle(
                annotated,
                (int(x / 2), int(y * 240 / 640)),
                (int((x + w) / 2), int((y + h) * 240 / 640)),
                (0, 255, 255),
                1,
            )
    sheet.paste(Image.fromarray(annotated), (0, row * 260))
    for col, (hmax, smin) in enumerate(((24, 55), (12, 45), (12, 90)), 1):
        mask = cv2.inRange(hsv, np.array((0, smin, 30)), np.array((hmax, 255, 255)))
        mask |= cv2.inRange(hsv, np.array((170, smin, 30)), np.array((179, 255, 255)))
        sheet.paste(Image.fromarray(mask).convert("RGB"), (col * 320, row * 260))
        draw.text((col * 320, row * 260 + 241), f"H<={hmax}, S>={smin}", fill="black")
out = Path("data/roboflow-dtr-v10-grouped/goal-mask-diagnostic.jpg")
sheet.save(out)
print(out)
