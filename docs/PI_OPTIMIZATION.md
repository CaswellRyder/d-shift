# Pi Zero W optimization experiments — 2026-10-05

Research integration only. No motor commands, flight approval, startup services,
package installs, or network changes. The original Pi demo and the original
[595-frame comparison](PI_COMPARISON.md) remain intact.

## Implemented changes

1. Retrained the same 6,263-parameter goal student for 25 epochs instead of eight.
   This is still a tiny 64x64 crop CNN distilled from a frozen MobileNetV4-Conv-Small
   teacher, **not** a MobileNetV4 running on the Pi. Same reviewed train/validation
   inputs, temperature 4, equal hard-label/KL weights, validation-loss checkpoint
   selection (epoch 18 of 25). Reserved test images were not used. Crop validation: FP32 93.97%,
   INT8 93.74%, versus the old FP32 80.60% / INT8 80.49%.
2. Added FP32 support to the native TensorFlow Lite C API adapter and `Predictor`.
   The deployed system library supports it without Python TensorFlow. The same
   eight-epoch network measured 40.31 ms/crop FP32 versus 54.68 ms INT8 on this
   actual ARMv6 board. FP32 costs 27,648 bytes versus 10,880 bytes INT8. Numerical
   format must be selected by device measurements, not assumed faster from size.
3. Avoided bounding-box calls for rejected contours and area calls for contours
   with fewer than three vertices. The final proposal lists matched the frozen
   implementation exactly on all 595 frames for both goals and balloons.
4. Added `TemporalVision`: bounded per-frame classification, oldest/unclassified
   candidates first, half-resolution local template matching with surrounding
   context, periodic full-image scans, explicit cached-label age/freshness,
   scene-change and target-loss handling. Cached labels expire; frame gaps and
   resolution changes discard tracks. No stale identity is intentionally retained
   across a failed association. Association is heuristic, not identity proof.

The slow-blimp tradeoff is to update **position** from each frame while reusing
**class identity** briefly. Tracking FPS is not fresh neural-classification FPS.
Default experimental intervals are 0.5 s full scan, 0.6 s class refresh, and a
1.0 s classification-age cutoff checked while assembling observations, including
elapsed inference work. This is not a sensor-to-controller frame-age guarantee. Inference budgets
must be reported: smaller budgets delay acquisition and can reduce recall.

The final experiment also allows 1.8 s refresh for predictions whose label is
background with score >=0.95. It never extends the accepted-label lifetime.
Appearance changes invalidate a cached classification, failed tracks are dropped,
and all tracks lost triggers another scan. A confidently misclassified target
with unchanged appearance can consequently take longer to be reconsidered.
Use `--background-refresh 0.6` to disable this extra amortization.

## Actual Pi: full-frame goal accuracy

Same 595 development frames, same threshold and proposal budget; no temporal reuse:

| Goal alternative | Mean processing | p95 | Yellow localization precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original eight-epoch INT8 | 761.43 ms | 863.79 ms | 38.39% | 58.91% | 46.48% |
| New 25-epoch FP32 | 609.00 ms | 1,021.12 ms | 78.30% | 90.06% | 83.77% |

Mean latency improved by 20.0%, but p95 worsened in this single full-frame run.
This is not a claim of uniform speedup or statistical significance. New Pi and
Mac full-frame TP/FP/FN counts matched exactly. New FP32 golden-crop outputs
matched all 21 reference labels with maximum score error 6.26e-7.

New **same-class** full-frame results are substantially weaker than crop accuracy:

| Class | Precision | Recall |
| --- | ---: | ---: |
| Orange circle | 14.17% | 14.00% |
| Orange square | 28.90% | 24.51% |
| Orange triangle | 81.01% | 31.84% |
| Yellow circle | 69.88% | 72.96% |
| Yellow square | 85.45% | 92.16% |
| Yellow triangle | 54.07% | 78.15% |

The new goal candidate is useful for integration experiments, **not qualified
competition perception**. Orange proposal/classification failures remain material.
The balloon model was not retrained in this optimization run.

## Actual Pi: controlled slow-motion replay

All rows use the same 240 generated frames. These are not live-camera FPS or
independent accuracy measurements. Yellow localization ignores shape.

| Method | Mean ms | p95 ms | Processing FPS | Yellow precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original RGB baseline | 76.87 | 88.99 | 13.01 | 0.00% | 0.00% |
| New FP32 student, stateless / 12 crops | 562.13 | 630.52 | 1.78 | 74.80% | 78.63% |
| Temporal / 4 crops, uniform refresh | 184.50 | 349.51 | 5.42 | 78.45% | 71.58% |
| Final temporal / 4 crops, slower high-score background refresh | 168.90 | 344.48 | 5.92 | 78.45% | 71.58% |

The final temporal policy is 3.33x faster than the new stateless network path on
this replay; its time is 2.20x the original color baseline's, not 10x. It costs
7.05 percentage points of yellow recall versus stateless. It performs 496 neural
calls instead of 2,592 (80.9% fewer), with 96 full scans instead of 240. These are
aggregate calls, not proof of a particular refresh rate for every tracked object.

The first three rows came from alternating-method runs. The final row is an
isolated repeat over the identical generated input, with external call timing
including observation assembly and duplicate suppression. A uniform-refresh
external-timer repeat measured 181.57 ms / 5.51 processing FPS; its yellow recall
was 71.15%. Wall-time-dependent label expiry can change a few observations between
runs. First-run temporal timing ended just before output assembly; use the final
external-timed run as the selected candidate's latency evidence.

The baseline's zero matches on these 12 selected source frames must not be hidden.
Its RGB/size configuration is uncalibrated for this public scene subset. This
does not establish that learned vision universally beats color segmentation.

Final replay: 0 false detections on the 24 inserted black frames, maximum emitted
accepted-label age 942.25 ms. At least one correctly classified target returned
on the first return frame in 11/12 clips, as did stateless; neither found a correct
target in the remaining clip. That is **not** proof every target was reacquired,
and full black frames are easier than partial occlusion or crossing objects.
158/240 processing calls exceeded the nominal 100 ms interval: this is not a
10 Hz real-time pass. The two-crop Mac-only ablation had 61.97% yellow recall,
so four crops was selected rather than maximizing speed at that recall cost.

## Actual live camera and checks

Selected policy: 60 real 320x240 OV5647 frames, **2.85 FPS measured full loop**,
310.60 ms mean processing, 396.90 ms p95, 3.22 processing FPS. All 60 sensor
timestamps increased. The uniform-refresh repeat was essentially the same at
2.86 FPS: the background optimization did **not** improve this camera scene.
Both used all 240 permitted neural calls; do not quote 5.92 FPS as live throughput.
No goal detections were accepted in these unlabeled frames, so the live trial
proves capture/runtime throughput only. First/last annotated PNGs are retained.

At the end, temperature was 44.4 C and `get_throttled` was `0x0`. The original
demo SHA remained `f46b1dfde183e4695a9daa00ff8d3e9ca01a5ab1c85e969b9992b332a58ce8a7`;
all 640 frozen comparison-bundle file checksums still matched. Local Ruff passed;
pytest: **177 passed, 2 skipped**. Native float/int tensor-buffer guards, cache
expiry, disappearance, reacquisition triggers, scene/resolution changes,
classification budgets, and background-refresh behavior have regression tests.

This is a selected **observational integration candidate**, not a flight release.
Next accuracy evidence must come from labeled, representative Pi-camera recordings
with neon-orange and yellow targets, lighting changes, occlusion, approach/retreat,
and slow rotations. The original method needs its own fair color calibration.

## Reproducibility and evidence boundaries

All new work is isolated in `/home/pacman/pi-optimization-20261005` on the Pi and
`runs/pi-optimization-20261005` in this repository. The new model is
`goal-long.float.tflite`, SHA-256
`b403ea6b94935d8d45fcc85256b997dc0bde9212a90c161780e224e962961ddf`.
Its original train/export receipts remain in `runs/goal-pi-student-long-20261005`.

Downloaded native reports, per-frame observations, model/code snapshots, camera
PNGs, and `verification.json` are in `runs/pi-optimization-results-20261005`:

- `crop-runtime.json`: same eight-epoch weights, INT8-versus-FP32 crop timing.
- `long-float-golden-pi.json`: new FP32 export numerical parity.
- `long-float-frames.json`: all 595 native full-frame results.
- `replay-budget4-pi.json`: original three-way controlled replay.
- `replay-budget4-retimed-pi.json`: full-call timer / uniform-refresh repeat.
- `replay-background-pi.json`: selected final temporal policy.
- `camera-background-pi.json`: selected policy's real camera throughput.

The companion `.frames.jsonl` files retain individual observations and timings.
`runs/pi-optimization-report-20261005.tar.gz` packages these results and runtime;
rerunning labeled benchmarks also requires the unchanged original comparison
bundle's 595 images and `pi_compare.py`. The current web viewer remains unchanged:
it is not automatically using this model or the temporal scheduler.

- Full-frame evaluation uses the original 595 lossless 320x240 development images,
  threshold 0.8, up to 12 proposals, nested duplicate rejection, IoU >=0.5 matching.
  It separately reports class-aware metrics and shape-agnostic yellow localization.
- The controlled temporal replay selects the first two unused frames containing
  each goal class: 12 clips, 20 frames each. Each clip translates a still by one
  pixel per nominal 100 ms, inserts two black missing-target frames, and returns
  the scene. These are **synthetic-motion engineering tests**, not independent
  recorded flight accuracy. All methods receive identical generated RGB frames.
- Replay timing excludes loading, camera, rendering, and writing. It records
  100 ms deadline overruns; a nominal clock does not mean a Pi achieves 10 Hz.
  Method order alternates, OpenCV and TFLite each use one thread, and warmups
  precede clips. The original baseline retains its original uncalibrated RGB
  and 200-pixel thresholds; it cannot classify shapes or orange goals.
- Live camera mode uses actual monotonic time and increasing sensor timestamps.
  It reports both processing throughput and measured capture-to-result loop
  throughput. Unlabeled desktop scenes establish no target-detection accuracy.
- Thresholds and training have used these development scenes repeatedly; they
  are not independent test-set evidence. Earlier YOLO results do not transfer.
- Goal boxes are not the goal's traversable opening. These experiments do not
  validate navigation, balloon capture, delivery, landing, or a motor watchdog.

## Run on the Pi

Use a new output filename on every run. Stop other camera owners first.

```bash
cd /home/pacman/pi-optimization-20261005
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 pi_temporal_bench.py \
  --base ../pi-comparison-20261005 --model goal-long.float.tflite \
  --mode camera --budget 4 --frames 100 --output camera-trial-02.json
```

This is bounded and observational. It saves first/last annotated PNGs, JSON
summary, and per-frame JSONL containing boxes, scores, track IDs, cache age,
freshness, scan decisions, and inference count. It does not open a web server.

```bash
python3 pi_optimize_bench.py --base ../pi-comparison-20261005 \
  --mode frames --model goal-long.float.tflite --output frames-repeat.json
python3 pi_temporal_bench.py --base ../pi-comparison-20261005 \
  --model goal-long.float.tflite --budget 4 --output replay-repeat.json
```

Keep `THIRD_PARTY_NOTICES.md` and the original dataset attribution with shared
images. Do not replace the original comparison with these tuned results.
