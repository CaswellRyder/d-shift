"""Calibrate a fisheye using checkerboards, in the flipped runtime image coordinates."""

import argparse
import json
import hashlib
from pathlib import Path

import cv2
import numpy as np

from dtr.lens import LensModel


def measured_fov(lens):
    w, h = lens.width, lens.height
    rays = lens.rays([[0, h / 2], [w, h / 2], [w / 2, 0], [w / 2, h], [0, 0], [w, h]])
    return [float(np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b)), np.dot(a, b))))
            for a, b in rays.reshape(3, 2, 3)]


def validate_candidate(K, D, size, rms, image_points):
    """Check every pixel and the continuous radial derivative over its ray range."""
    if not np.isfinite(rms) or rms < 0:
        raise ValueError("Invalid calibration RMS")
    lens = LensModel(K, D, size, True, rms)
    x, y = np.meshgrid(np.arange(size[0]), np.arange(size[1]))
    pixels = np.column_stack((x.ravel(), y.ravel())).astype(np.float64)
    rays = lens.rays(pixels)
    normalized = (rays[:, :2] / rays[:, 2:]).reshape(-1, 1, 2)
    projected = cv2.fisheye.distortPoints(normalized, lens.K, lens.D).reshape(-1, 2)
    error = np.linalg.norm(projected - pixels, axis=1)
    if not np.isfinite(error).all() or error.max() > 1e-5:
        raise ValueError("Full-image fisheye roundtrip failed")
    observed = np.concatenate(image_points, axis=1).reshape(-1, 2)
    observed_rays = lens.rays(observed)
    theta_max = float(np.arccos(np.clip(rays[:, 2], -1, 1)).max())
    observed_max = float(np.arccos(np.clip(observed_rays[:, 2], -1, 1)).max())
    # d(theta_d)/d(theta) = 1 + 3*k1*t + 5*k2*t² + 7*k3*t³ + 9*k4*t⁴.
    derivative = np.polynomial.Polynomial(np.r_[1., D.ravel() * [3, 5, 7, 9]])
    roots = derivative.deriv().roots()
    bounds = [0., theta_max ** 2]
    bounds.extend(float(r.real) for r in roots
                  if abs(r.imag) < 1e-10 and 0 < r.real < theta_max ** 2)
    minimum = float(min(derivative(np.asarray(bounds))))
    if not np.isfinite(minimum) or minimum <= 0:
        raise ValueError("Non-monotonic fisheye radial mapping")
    return dict(full_grid_pixels=len(pixels), full_grid_inverse_valid=len(pixels),
                full_grid_roundtrip_max_px=float(error.max()),
                roundtrip_tolerance_px=1e-5, minimum_radial_derivative=minimum,
                radial_checked_theta_max_rad=theta_max,
                observed_theta_max_rad=observed_max,
                corner_bounds_px=[observed.min(axis=0).tolist(), observed.max(axis=0).tolist()],
                status="numerical checks only; no independent physical validation")


def calibrate(folder, cols, rows, square_size, output, *, flip=True, size=(320, 240),
              input_flipped=False, distortion_terms=4, seed_focal=None):
    if input_flipped and flip:
        raise ValueError('Already-flipped runtime frames must not be flipped a second time')
    if min(cols, rows) < 2 or not np.isfinite(square_size) or square_size <= 0 or min(size) <= 0:
        raise ValueError("Invalid board or image dimensions")
    if distortion_terms not in (1, 2, 4):
        raise ValueError("distortion_terms must be 1, 2 or 4")
    if seed_focal is not None and (not np.isfinite(seed_focal) or seed_focal <= 0):
        raise ValueError("Seed focal length must be finite and positive")
    output = Path(output)
    if output.exists():
        raise ValueError("Output exists; choose a new candidate path")
    folders = [folder] if isinstance(folder, (str, Path)) else list(folder)
    paths = sorted(path for source in folders for path in Path(source).iterdir()
                   if path.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("Duplicate input images")
    inputs = []
    objects = np.zeros((1, cols * rows, 3), np.float64)
    objects[0, :, :2] = np.mgrid[:cols, :rows].T.reshape(-1, 2) * square_size
    object_points, image_points = [], []
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-8)
    for path in paths:
        if path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
            continue
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Unreadable input image: {path}")
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        if flip:
            image = cv2.flip(image, -1)
        found, corners = cv2.findChessboardCorners(
            image, (cols, rows), cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        inputs.append(dict(path=str(path.resolve()),
                           sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                           board_detected=bool(found)))
        if found:
            corners = cv2.cornerSubPix(image, corners, (3, 3), (-1, -1), criteria)
            object_points.append(objects.copy())
            image_points.append(corners.astype(np.float64).reshape(1, -1, 2))
    if len(image_points) < 10:
        raise ValueError(f"Need at least 10 detected checkerboards; found {len(image_points)}")
    flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW
    fixed = list(range(distortion_terms + 1, 5))
    for coefficient in fixed:
        flags |= getattr(cv2.fisheye, f"CALIB_FIX_K{coefficient}")
    initial_K = np.zeros((3, 3))
    if seed_focal is not None:
        initial_K = np.array([[seed_focal, 0, (size[0] - 1) / 2],
                              [0, seed_focal, (size[1] - 1) / 2], [0, 0, 1.]])
        flags |= cv2.fisheye.CALIB_USE_INTRINSIC_GUESS
    rms, K, D, _, _ = cv2.fisheye.calibrate(
        object_points, image_points, size, initial_K, np.zeros((4, 1)),
        flags=flags,
        criteria=criteria)
    validation = validate_candidate(K, D, size, rms, image_points)
    lens = LensModel(K, D, size, True, rms)
    fov = measured_fov(lens)
    data = dict(K=K.tolist(), D=D.ravel().tolist(), image_size=list(size),
                rms=float(rms), flipped=bool(flip or input_flipped),
                input_flipped=bool(input_flipped), images_used=len(image_points),
                distortion_terms=distortion_terms, seed_focal_px=seed_focal,
                fixed_distortion_coefficients=[f"k{i}" for i in fixed],
                board_inner_corners=[cols, rows], square_size_m=square_size,
                source_folders=[str(Path(source).resolve()) for source in folders],
                inputs=inputs, opencv_version=cv2.__version__, calibration_flags=flags,
                validation=validation, status="provisional; not activated",
                limitations=["No independent lower-half or lower-corner physical validation.",
                             "Numerical checks do not establish flight qualification.",
                             "Radial symmetry is a model assumption; no mirrored views added."] )
    payload = json.dumps(data, indent=2, allow_nan=False) + "\n"
    with output.open("x") as stream:
        stream.write(payload)
    print(f"RMS: {rms:.6f}; images used: {len(image_points)}")
    print(f"FOV degrees: horizontal {fov[0]:.3f}, vertical {fov[1]:.3f}, diagonal {fov[2]:.3f}")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", nargs="+")
    parser.add_argument("--distortion-terms", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument("--seed-focal", type=float, help="initial fx=fy in pixels")
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
                  input_flipped=args.input_flipped, distortion_terms=args.distortion_terms,
                  seed_focal=args.seed_focal)
    except (ValueError, OSError, cv2.error) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
