# Current balloon students: original Pi hardware evidence

2026-10-08. SSH restored using password authentication after the user's reboot.
A fresh password login also succeeded after the trials. Non-interactive key-only
authentication still fails; no keys, SSH configuration, network configuration or
system services were changed. A reboot alone does not establish an authentication fix.

## Measurements

Six sequential trials on Raspberry Pi Zero W Rev 1.1, ARMv6, OpenCV 4.10.0,
native TensorFlow Lite C runtime, one inference/OpenCV thread. Each trial loaded
one student and one search method, checked all 16 frames against desktop golden
predictions during warmup, then processed two timed repetitions. No competing
vision implementation was run. Ordinary desktop/system services stayed enabled;
this is not a fully isolated operating system.

Pooled processing timings from the raw frame records:

- Seed 42, HSV search: **3.878 FPS**, mean 257.861 ms, p95 372.716 ms.
- Seed 42, MSER search: **2.369 FPS**, mean 422.119 ms, p95 810.218 ms.
- Seed 43, MSER search: **2.356 FPS**, mean 424.495 ms, p95 820.318 ms.

Each entry pools 64 timing samples across two trials, but only **16 unique
development images**. Percentiles were recomputed from pooled records, not
averaged across reports. HSV repeat B includes an 819.628 ms outlier and lower
CPU/wall utilization; it remains in the results. Trial order was seed42 HSV A,
MSER A, MSER B, HSV B, then seed43 MSER A and B.

These are warmed, 320x240 full-scene search plus 64x64 crop inference and selection
measurements. They exclude camera capture, file loading, model loading, warmup
and golden verification overhead. They are not live-camera FPS. This HSV baseline
uses the same neural model and research part suppression as MSER: it is **not the
other team's pixel-only implementation** or an unchanged production baseline.

Seed42 MSER spends about 314 ms on search, 107 ms on neural inference and 1 ms
on selection. It makes 5.625 neural calls per frame versus HSV's 8.75, but its
more expensive search dominates. MSER process CPU/wall ratios were approximately
99.3–99.5%; maximum process RSS was about 112 MiB. All recorded throttle flags
were `0x0`; recorded temperatures ranged approximately 42.8–47.1 C.

## Prediction parity and accuracy limitations

All six trials passed desktop/Pi comparison of proposal geometry, crop boxes,
class, acceptance and suppression. Maximum score difference was below 0.000001
(tolerance 0.001), despite desktop OpenCV 4.11.0 versus Pi 4.10.0.

The 12-image indoor development panel contains 11 red and 12 blue annotations.
Seed42 HSV yields red TP/FP/FN **9/0/2**, blue **8/1/4**. MSER yields red
**11/0/0**, blue **9/4/3**: better recall here, but more blue false positives.
Seed43 MSER yields red **10/2/1**, blue **8/4/4**; it is not a stronger replacement.

On the separate four-image original development panel, seed42 MSER recovers the
third blue target, but retains two red false positives. Full per-panel counts
are in `current-pi-search-measurements.json`. Repetitions do not increase the
accuracy sample size. These small, repeatedly inspected development panels do
not establish 94% precision/recall or flight readiness. No reserved final test
was evaluated and no deployment was promoted.

## Separate search profile

After all normal timing trials, `profile_balloon_search.py` profiled search only
on the same 16 frames, without loading a neural model. The instrumented total
was 5.723 seconds, including:

- MSER `detectRegions`: 1.903 seconds, 32 calls.
- `convexHull`: 1.432 seconds, 773 calls.
- NumPy region means: 0.692 seconds cumulative, 816 calls.

Profile timings include instrumentation and are not a replacement FPS result.
Nested cumulative times must not be added together. The Pi profile establishes
both region detection and hull computation as substantial targets; the desktop
profile alone would overemphasize hull computation.

Next bounded experiment: reject impossible bounding geometry before hull work,
then test an exact hull-preserving reduction of region points. Require unchanged
proposal geometry/scores and full-scene predictions, followed by fresh sequential
Pi timings. No speedup from that proposed optimization is claimed yet.

## Reproduction and identities

Code checkpoint: `9a82d891`. Ignored host bundle:
`artifacts/balloon-pi-current-20261008`; isolated Pi directory:
`/home/pacman/balloon-pi-current-20261008`. Existing deployment directories were
not replaced. Raw reports/logs are retained locally under
`artifacts/balloon-pi-current-results-20261008`.

Models `new-views-42` and `new-views-43` are 7,763-parameter, 35,204-byte FP32
separable-context students from the MobileNetV4-Conv-Small teacher lineage, not
full MobileNetV4 inference on the Pi. Model hashes, bundle/input/golden hashes,
raw-log hashes, health observations and each trial are retained in
`current-pi-search-measurements.json`. The selected existing runtime was:

`/home/pacman/dtr-runtime-build-20261006/libtensorflowlite_c.candidate.so`

SHA-256: `54ed7fdda0b5b5f6db05a538eb800b7bc7ecd70d32f67dc13d1e78817d8b653a`.

Example command from the isolated Pi bundle directory:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 DTR_TFLITE_LIBRARY=/home/pacman/dtr-runtime-build-20261006/libtensorflowlite_c.candidate.so python3 pi_balloon_search_bench.py --base . --model new-views-42 --search mser --rounds 2 --output results/fresh-trial.json
```

Use a fresh output filename. Run methods sequentially. No camera was opened in
this step, and no ESP32, serial or motor commands were issued. Live camera
orientation, capture latency, range calibration and independent field
qualification remain separate pending work.
