"""Calibrate a fisheye using checkerboards, in the flipped runtime image coordinates."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from dtr.lens import LensModel


def measured_fov(lens):
    w, h = lens.width, lens.height
    rays = lens.rays([[0, h / 2], [w, h / 2], [w / 2, 0], [w / 2, h], [0, 0], [w, h]])
    return [float(np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b)), np.dot(a, b))))
            for a, b in rays.reshape(3, 2, 3)]


def calibrate(folder, cols, rows, square_size, output, *, flip=True, size=(320, 240),
              input_flipped=False):
    if input_flipped and flip:
        raise ValueError('Already-flipped runtime frames must not be flipped a second time')
    if min(cols, rows) < 2 or not np.isfinite(square_size) or square_size <= 0 or min(size) <= 0:
        raise ValueError("Invalid board or image dimensions")
    objects = np.zeros((1, cols * rows, 3), np.float64)
    objects[0, :, :2] = np.mgrid[:cols, :rows].T.reshape(-1, 2) * square_size
    object_points, image_points = [], []
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-8)
    for path in sorted(Path(folder).iterdir()):
        if path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
            continue
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        if flip:
            image = cv2.flip(image, -1)
        found, corners = cv2.findChessboardCorners(
            image, (cols, rows), cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if found:
            corners = cv2.cornerSubPix(image, corners, (3, 3), (-1, -1), criteria)
            object_points.append(objects.copy())
            image_points.append(corners.astype(np.float64).reshape(1, -1, 2))
    if len(image_points) < 10:
        raise ValueError(f"Need at least 10 detected checkerboards; found {len(image_points)}")
    rms, K, D, _, _ = cv2.fisheye.calibrate(
        object_points, image_points, size, np.zeros((3, 3)), np.zeros((4, 1)),
        flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW,
        criteria=criteria)
    lens = LensModel(K, D, size, True, rms)
    fov = measured_fov(lens)
    data = dict(K=K.tolist(), D=D.ravel().tolist(), image_size=list(size),
                rms=float(rms), flipped=bool(flip or input_flipped),
                input_flipped=bool(input_flipped), images_used=len(image_points))
    Path(output).write_text(json.dumps(data, indent=2) + "\n")
    print(f"RMS: {rms:.6f}; images used: {len(image_points)}")
    print(f"FOV degrees: horizontal {fov[0]:.3f}, vertical {fov[1]:.3f}, diagonal {fov[2]:.3f}")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder")
    parser.add_argument("--cols", type=int, required=True, help="inner corner columns")
    parser.add_argument("--rows", type=int, required=True, help="inner corner rows")
    parser.add_argument("--square-size", type=float, required=True, help="meters")
    parser.add_argument("--flip", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--input-flipped', action='store_true',
                        help='Raw lab frames already have the runtime 180-degree rotation; do not flip again')
    parser.add_argument("--size", default="320x240")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        size = tuple(map(int, args.size.lower().split("x")))
        if len(size) != 2:
            raise ValueError("Expected --size WIDTHxHEIGHT")
        calibrate(args.folder, args.cols, args.rows, args.square_size, args.output,
                  flip=args.flip and not args.input_flipped, size=size,
                  input_flipped=args.input_flipped)
    except (ValueError, cv2.error) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
