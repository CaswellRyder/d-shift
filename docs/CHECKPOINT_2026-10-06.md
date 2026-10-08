# Checkpoint before further Pi-only work — October 6, 2026

User requested this bookmark before proceeding. ESP32 hardware is not ready;
do not attach it or issue motor/serial commands during this phase.

## Preserved baseline

- Hardware: original Raspberry Pi Zero W, ARMv6; Pi camera.
- Neural deployment: custom context FP32 student distilled from MobileNetV4,
  not the full MobileNetV4 teacher; rebuilt TFLite runtime selected per process.
- Model SHA-256: `147671bbff83a8d0f57ff01a0fc8836441790b7b5c919f0e1ade1bb16c75fea3`.
- Runtime SHA-256: `54ed7fdda0b5b5f6db05a538eb800b7bc7ecd70d32f67dc13d1e78817d8b653a`.
- Full isolated evidence: `/Volumes/HDD_Storage_Extended/dtr-isolated-tests-20261006/REPORT.md`.
- Upright-camera evidence: `/Volumes/HDD_Storage_Extended/dtr-upright-20261006/pi-results/`.
- Pi upright runner: `/home/pacman/upright-comparison-20261006/isolated_bench.py`.
- Pixel implementation: `/home/pacman/Documents/TESTING/hollow_test_again.py`.
  Its pre-rotation original remains in `hollow_test_again.py.pre-upright-20261006`.

## Last full isolated comparison (before camera rotation)

| Measurement | Pixels | Context hybrid |
|---|---:|---:|
| Live camera FPS | 2.58 | 3.36 |
| Paced replay delivered FPS | 9.87 | 8.29 |
| Static yellow precision | 91.53% | 88.64% |
| Static yellow recall | 10.13% | 90.81% |
| Images with any correct yellow goal | 54/256 | 254/256 |

Live search resolutions differ: pixels 640x480, context 320x240. Live footage
was an unlabeled desk; no arena accuracy or control deadline was established.
Static results use 595 development images and the heavier stateless policy.
Pixels selects one goal; context can return multiple goals.

Post-hoc size analysis at 320x240, size = square root of labeled box area:
under 16px: pixels 0/107, context 99/107; 16–32px: pixels 29/323,
context 282/323. These are not measured-distance or fast-live-policy results.

Both camera paths now rotate 180 degrees before detection; isolated smoke checks
and visual inspections passed. No complete post-rotation benchmark exists at
this checkpoint. No thresholds or weights changed for the orientation fix.

## Next bounded work

1. Check Pi availability and competing camera/process workloads.
2. Rerun separate upright-camera timing trials, retaining old receipts.
3. Analyze the fast temporal replay by small-target coverage and reacquisition,
   separating dropped frames from processed-frame misses.
4. Improve reproducible Pi-only validation before changing models or thresholds.

Actual marked-distance/color trials need staged physical targets. Distance
calibration, balloon capture confirmation, and ESP32 integration remain unproven.
