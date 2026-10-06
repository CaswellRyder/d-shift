"""Build or exhaustively verify an exact color lookup, with no training/test images."""
import argparse
from pathlib import Path

import cv2
import numpy as np

from dtr.data import sha256, write_json
from dtr.color_lookup import GoalColorLookup
from dtr.vision import color_masks


def rgb_slice(red):
    rgb = np.empty((256,256,3),np.uint8)
    rgb[:,:,0] = red
    rgb[:,:,1] = np.arange(256,dtype=np.uint8)[:,None]
    rgb[:,:,2] = np.arange(256,dtype=np.uint8)[None,:]
    return rgb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--verify", action="store_true", help="Check all 16,777,216 colors with installed OpenCV")
    args = parser.parse_args()
    path = Path(args.output)
    cv2.setNumThreads(1)
    if args.verify:
        lookup = GoalColorLookup(path)
        for red in range(256):
            rgb = rgb_slice(red)
            expected = color_masks(rgb,"goal","balloon_components")
            assert all(np.array_equal(a,b) for a,b in zip(expected,lookup.masks(rgb))), f"Mismatch at red {red}"
        print(dict(colors_verified=1<<24, opencv=cv2.__version__, sha256=sha256(path)),flush=True)
        return
    if path.exists() or path.with_suffix(".json").exists():
        raise FileExistsError(path)
    values = np.empty((256,256,256),np.uint8)
    for red in range(256):
        orange,yellow = color_masks(rgb_slice(red),"goal","balloon_components")
        values[red] = (orange//255) | ((yellow//255)<<1)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("xb") as stream:
        stream.write(values.tobytes())
    write_json(path.with_suffix(".json"),dict(contract="dtr-goal-rgb24-hsv-v1",sha256=sha256(path),
        bytes=path.stat().st_size,opencv=cv2.__version__,vision_source_sha256=sha256("src/dtr/vision.py"),
        color_order="RGB, index=(R<<16)|(G<<8)|B",bits="orange=1 yellow=2",synthetic=False,
        deployment_approved=False))
    print(dict(path=str(path),bytes=path.stat().st_size),flush=True)


if __name__ == "__main__":
    main()
