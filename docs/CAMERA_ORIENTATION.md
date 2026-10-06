# Pi camera orientation — 2026-10-06

The physical camera is mounted upside down. Apply a 180-degree rotation at camera
configuration, before either detection or display, using
`libcamera.Transform(hflip=True, vflip=True)`. Both axes together rotate the image;
this is not a mirror-only preview correction or a per-frame Python image copy.

## Updated entry points

- Repository scripts `competition_camera.py`, `pi_temporal_bench.py`, and
  `pi_live_compare.py` in `scripts/` default to `--camera-rotation 180`.
  Use `--camera-rotation 0` for an upright physical mount.
- The new Pi isolated runner is
  `/home/pacman/upright-comparison-20261006/isolated_bench.py`, with the same option.
  Both methods use its corrected capture path in separate test processes.
- The current standalone pixel implementation on the Pi is
  `/home/pacman/Documents/TESTING/hollow_test_again.py`. Its camera generator now
  uses `CAMERA_ROTATION = 180`. Set this constant to `0` to disable the correction.
  The original is backed up alongside it as
  `hollow_test_again.py.pre-upright-20261006`.

Mac webcam, upload, and replay inputs are not automatically rotated. Old frozen Pi
comparison bundles remain unchanged; use the new upright runner for new camera
trials. Detector thresholds and model weights were not changed by this fix.

Coordinates now refer to orientation-corrected camera pixels. This correction
does not establish camera calibration, world heading, or flight-control signs.

## Verification

- Sequential, separate-process Pi camera smoke tests for pixel and context methods
  both reported the configured `hvflip` transform and passed isolation checks.
- Captured the patched standalone pixel generator's output and closed the camera.
- Visually inspected the context and standalone pixel frames for upright orientation.
- Two rotation/cleanup tests passed for the standalone generator (0 and 180 degrees).
- Eleven existing comparison tests passed; lint passed for the three repository scripts.
- Camera processes were stopped afterward; no motor commands were sent.

These short checks validate orientation, not new accuracy or reliable FPS results.
Previous isolated performance reports remain historical measurements from before
this correction; the complete benchmark was not rerun for this change.

Saved frames and reports on the Mac:
`/Volumes/HDD_Storage_Extended/dtr-upright-20261006/pi-results/`.
