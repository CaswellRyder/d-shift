# Optimized balloon search: live camera plumbing check

2026-10-08. The original Pi Zero W camera works, but the observed scene was
nearly black. **This is camera-path evidence, not representative balloon
throughput, detection accuracy, visual orientation confirmation or flight
qualification.** No deployment was changed.

## Hardware and configuration

The camera enumerated as OV5647, 2592x1944, 10-bit. The bounded run explicitly
selected that full sensor mode with 320x240 RGB888 main output, requested 10 FPS,
two buffers and `queue=False`. It verified the configured sensor dimensions,
format and horizontal-plus-vertical flip (180 degrees). Source orientation was
not visually confirmable from the dark image.

The pipeline converts camera RGB888/BGR memory into owned RGB pixels before
releasing the camera request. The request has a three-second wait timeout;
collection is bounded by frame count and a soft wall-clock limit, with an
external 120-second process timeout. The wall limit can extend by one in-progress
frame. The camera is closed in a `finally` block. No model or image loading and no
PNG/JSON writes occur inside the timed capture loop. First/last samples remain
local in ignored artifacts, not in version control.

These API choices were checked against the installed Picamera2 API and Raspberry
Pi's [Picamera2 implementation](https://github.com/raspberrypi/picamera2/blob/main/picamera2/picamera2.py)
and [pixel format mapping](https://github.com/raspberrypi/picamera2/blob/main/picamera2/request.py).
Sensor timestamps are compared against Linux CLOCK_BOOTTIME. Missing, negative,
future and repeated timestamps are not silently counted as valid freshness
evidence. Valid ages do not establish an acceptable flight-control deadline.

## Observed run

Student `new-views-42`, optimized MSER, unchanged existing ARMv6 native TFLite
runtime and the helper from `EXACT_SEARCH_OPTIMIZATION.md`. All 16 development
scenes passed the old-golden check before camera capture. No other vision
implementation ran concurrently.

- 40 frames in 8.012 seconds: **4.992 live FPS in this dark scene only**.
- Mean processing: 161.600 ms, almost entirely search.
- **Zero neural calls and zero accepted red/blue detections.** This was not a
  loaded neural-inference performance test.
- Mean capture/wait/conversion: 38.188 ms; mean frame wall time 199.937 ms.
- Sensor-to-result: mean **283.445 ms**, p95 **294.151 ms**, maximum 309.085 ms.
- 40/40 valid and strictly increasing sensor timestamps; none omitted.
- Frame metadata reported 99,978 microseconds per sensor frame (about 10 FPS),
  while processed timestamps advanced about 200 ms. Not every sensor frame was
  processed; no assumption that requested FPS equals processing FPS.
- Exposure metadata: 66,663 microseconds. Sensor-to-acquire age was approximately
  120–140 ms even with `queue=False`, so disabling retained frames does not
  eliminate sensor/ISP pipeline latency.
- Maximum process RSS 122,712 KiB (119.84 MiB); throttle flags `0x0` before/after.

Do not replace the earlier **2.991 FPS loaded replay** result with this roughly
5 FPS number. The workloads are different. The loaded replay still supplies
the valid paired old-versus-optimized speed comparison.

## Why content qualification remains open

The first sample's mean RGB value was 1.02/255; the last was 1.30/255. Respectively
99.90% and 99.35% of pixels had all channels below 12. Both fail the added
nearly-black diagnostic (99% of pixels below that threshold). This is a
brightness warning, **not a diagnosis of a lens cap or camera fault**. Only
first/last images were saved, so those statistics do not label all intermediate
frames. Zero detections are not validated true negatives.

The user was asked to uncover/reposition the camera toward a lit scene, ideally
with red and blue balloons. A new bounded run can then exercise actual proposals,
check visible orientation, and retain samples for review. Labeled representative
scene coverage and physical range validation remain necessary for qualification.
No ESP32, serial, actuator or motor commands were issued.

## Artifacts and repeat command

- Capturing tool: `scripts/pi_balloon_live_bench.py`.
- Artifact verifier: `scripts/review_balloon_live.py`. It verifies all retained
  file hashes, recomputes timestamp validity/ages/rate, and checks first/last
  brightness. It does not assign balloon labels or promote a model.
- Tracked evidence: `fast-search-live-dark-scene.json`.
- Ignored raw output: `artifacts/balloon-live-fast-a-20261008`.
- Pi output: `/home/pacman/balloon-pi-fast-search-20261008/live-fast-a`.
- Capturing script SHA-256:
  `c82ed3506a77972f3ac257102547f618fb232c8ce7e8708e9790a8ef12153146`.

The camera tool was added alongside the immutable research bundle; it has its
own recorded script hash. It did not replace any bound bundle file.

From the Pi research bundle directory, with a **new** output name and no other
camera/vision process running:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 DTR_TFLITE_LIBRARY=/home/pacman/dtr-runtime-build-20261006/libtensorflowlite_c.candidate.so timeout --signal=TERM --kill-after=5s 120s python3 -u pi_balloon_live_bench.py --base . --region-library native/regions.so --count 40 --seconds 30 --fps 10 --output live-lit-a
```

Review the copied artifacts on the host with `scripts.review_balloon_live`,
passing `--input` and a fresh `--output`. Do not use new live footage as training
data automatically: label/provenance and split decisions must be explicit.
