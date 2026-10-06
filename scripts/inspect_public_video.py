"""Extract sparse evidence frames; no annotations or training claims are generated."""

import argparse
from pathlib import Path

import cv2
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print({"fps": fps, "frames": count, "seconds": count / fps})
    sheet = Image.new("RGB", (960, 4 * 265), "white")
    for i in range(12):
        index = int(i * (count - 1) / 11)
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            continue
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image.save(args.output / f"frame-{index:06d}.png")
        image.thumbnail((320, 240))
        x, y = (i % 3) * 320, (i // 3) * 265
        sheet.paste(image, (x, y + 25))
        ImageDraw.Draw(sheet).text(
            (x + 5, y + 5), f"frame {index}; {index / fps:.2f}s", fill="black"
        )
    cap.release()
    sheet.save(args.output / "contact-sheet.jpg")


if __name__ == "__main__":
    main()
