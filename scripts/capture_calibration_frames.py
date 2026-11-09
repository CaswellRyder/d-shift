"""Save checkerboard frames for scripts/calibrate_lens.py, from the flight camera.

Uses the flight loop's Camera (320x240, 180-degree runtime flip), so calibrate the
result with --input-flipped. Only frames where the board is found are kept, at most
one per --every seconds, and each save prints where the board sat in the frame so the
operator can be told to cover the centre, the edges and the corners.
"""
import argparse
from pathlib import Path
import time

import cv2

from pi_flight_loop import Camera


def region(corners, width=320, height=240):
    x, y = corners[:, 0, 0].mean(), corners[:, 0, 1].mean()
    col = "left" if x < width / 3 else "right" if x > 2 * width / 3 else "centre"
    row = "top" if y < height / 3 else "bottom" if y > 2 * height / 3 else "middle"
    span = corners[:, 0, 0].max() - corners[:, 0, 0].min()
    return f"{row}-{col}", int(span)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="New folder for the frames")
    parser.add_argument("--cols", type=int, default=7, help="inner corner columns")
    parser.add_argument("--rows", type=int, default=4, help="inner corner rows")
    parser.add_argument("--seconds", type=float, default=90)
    parser.add_argument("--every", type=float, default=1.5, help="Minimum seconds between saves")
    parser.add_argument("--max-frames", type=int, default=40)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True)
    camera = Camera(10)
    saved, last, seen = 0, 0.0, {}
    end = time.monotonic() + args.seconds
    try:
        while time.monotonic() < end and saved < args.max_frames:
            rgb, now = camera.read()
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            found, corners = cv2.findChessboardCorners(
                gray, (args.cols, args.rows),
                cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK)
            if not found or now - last < args.every:
                continue
            last = now
            cv2.imwrite(str(out / f"{saved:03d}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            where, span = region(corners)
            seen[where] = seen.get(where, 0) + 1
            saved += 1
            print(f"saved {saved:3d}  board at {where:14s} {span:3d} px wide", flush=True)
    finally:
        camera.close()
    print(f"done: {saved} frames in {out}; coverage {seen}", flush=True)


if __name__ == "__main__":
    main()
