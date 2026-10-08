"""Exact red/blue RGB24 lookup; exhaustive verification uses no dataset images."""
import argparse
import os
from pathlib import Path
import platform
import time

import cv2
import numpy as np

from dtr.color_lookup import RedBlueColorLookup
from dtr.data import sha256, write_json
from dtr.vision import RED_BLUE_PROFILE, color_masks


def rgb_slice(red):
    rgb = np.empty((256, 256, 3), np.uint8)
    rgb[:, :, 0] = red
    rgb[:, :, 1] = np.arange(256, dtype=np.uint8)[:, None]
    rgb[:, :, 2] = np.arange(256, dtype=np.uint8)[None, :]
    return rgb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--native-library", type=Path, help="Verify target-built C backend instead of NumPy")
    args = parser.parse_args()
    # Reference must always execute OpenCV, never accidentally compare LUT to itself.
    os.environ.pop("DTR_RED_BLUE_COLOR_LOOKUP", None)
    os.environ.pop("DTR_RED_BLUE_LOOKUP_LIBRARY", None)
    if args.native_library:
        if not args.verify:
            parser.error("Native backend is only used for verification")
        os.environ["DTR_RED_BLUE_LOOKUP_LIBRARY"] = str(args.native_library.resolve())
    cv2.setNumThreads(1)
    path = args.output
    if args.receipt and args.receipt.exists():
        raise FileExistsError(args.receipt)
    started = time.perf_counter()
    if args.verify:
        lookup = RedBlueColorLookup(path)
        for red in range(256):
            rgb = rgb_slice(red)
            expected = color_masks(rgb, "balloon", RED_BLUE_PROFILE)
            if not all(np.array_equal(a, b) for a, b in zip(expected, lookup.masks(rgb))):
                raise ValueError(f"Mask mismatch at red={red}")
        result = dict(colors_verified=1 << 24, opencv=cv2.__version__, machine=platform.machine(),
                      sha256=sha256(path), seconds=time.perf_counter()-started,
                      native_library_sha256=sha256(args.native_library) if args.native_library else None,
                      scope="Exact color-mask equivalence, not detection accuracy", deployment_approved=False)
        if args.receipt:
            write_json(args.receipt, result)
        print(result, flush=True)
        return
    if path.exists() or path.with_suffix(".json").exists():
        raise FileExistsError(path)
    values = np.empty((256, 256, 256), np.uint8)
    for red in range(256):
        a, b = color_masks(rgb_slice(red), "balloon", RED_BLUE_PROFILE)
        values[red] = (a//255) | ((b//255) << 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(values.tobytes())
    write_json(path.with_suffix(".json"), dict(contract=RedBlueColorLookup.contract, sha256=sha256(path),
               bytes=path.stat().st_size, opencv=cv2.__version__, vision_source_sha256=sha256("src/dtr/vision.py"),
               bits="red=1 blue=2", color_order="RGB, index=(R<<16)|(G<<8)|B", deployment_approved=False))
    print(dict(path=str(path), bytes=path.stat().st_size), flush=True)


if __name__ == "__main__":
    main()
