# Updated pixel method versus context student — October 6, 2026

Status: completed on the actual Pi. No benchmark or camera process was left running.

**Verdict:** the updated pixel detector is conservative and fast on common
320×240 inputs, but misses substantially more yellow goals with its supplied
settings. The context model is the stronger detection candidate in these tests.
With temporal reuse it delivered 7.93 FPS in paced replay; in the actual desk-camera
scene it delivered 3.31 FPS versus the pixel configuration's 2.59 FPS. These
measurements are not a competition qualification or a prediction of arena FPS.

## Common-input static results — actual Pi

| Yellow localization, 595 images | Updated hollow pixel method | Context FP32 + rebuilt runtime |
| --- | ---: | ---: |
| True positives / false positives / misses | 54 / 5 / 479 | 484 / 62 / 49 |
| Precision | 91.53% | 88.64% |
| Recall | 10.13% | 90.81% |
| F1 | 18.24% | 89.71% |
| Target-containing frames with any correct goal | 54/256 (21.09%) | 254/256 (99.22%) |
| Mean processing | 96.71 ms | 382.12 ms |
| p95 processing | 182.98 ms | 445.09 ms |
| Stateless processing throughput | 10.34 FPS | 2.62 FPS |

The new pixel method is conservative: few false positives, but substantial misses
in this configuration. Its one-goal output partly limits all-target recall, but
does not explain the low any-correct-goal frame rate. The model finds far more
goals at approximately 3.95 times the stateless processing cost. This does not
describe the temporal path's throughput, and these are not native 640×480 pixel
accuracy results. Neither method was recalibrated on these evaluation frames.

## Wall-clock-paced replay — actual Pi

| Metric | Updated hollow pixel method | Context temporal + rebuilt runtime |
| --- | ---: | ---: |
| Available source frames | 720 | 720 |
| Processed / skipped frames | 702 / 18 | 581 / 139 |
| Delivered result FPS | 9.85 | 7.93 |
| Processing-only throughput | 13.92 FPS | 8.94 FPS |
| Mean / p95 processing | 71.82 / 131.33 ms | 111.81 / 236.12 ms |
| p95 source-frame-to-result age | 157.53 ms | 241.52 ms |
| Processed target frames with any correct yellow goal | 292/522 (55.94%) | 429/429 (100%) |
| Processed-frame yellow precision / recall | 96.05% / 21.07% | 96.59% / 74.98% |
| Detections on processed black frames | 0 | 0 |

The temporal model is approximately 1.24 times slower by delivered-rate ratio
in this particular replay, not ten times slower. Frame skipping matters: its
8.94 processing FPS becomes 7.93 delivered FPS under the 10 FPS source schedule.
Neither implementation is a strict 100 ms deadline pass. Result age here is
relative to the generated source timeline, not a physical camera timestamp.

Different processed temporal samples and a small, correlated twelve-still input
set prevent treating the replay percentages as an independent accuracy victory.
Both repeats, all method outputs, timings, truth transformations, and skipped
indices are retained in `paced-v2.jsonl`.

After the blackout, at least one correct yellow goal reappeared in the output
for 16/20 eligible pixel clip-repeats and 20/20 context clip-repeats. Among those
successful recoveries, first-result delay from the programmed return ranged
83–1,240 ms for pixels and 207–241 ms for context. Four pixel clip-repeats never
recovered a correct yellow goal before the clip ended. These counts do not prove
identity continuity or recovery of every individual goal; repeats share images.

## Live camera results — actual Pi, native processing sizes

The saved raw frame shows a dim, tilted desk/equipment scene, not an arena with
staged goals. Both methods emitted zero yellow boxes. This is a runtime/capture
test only; there is no labeled live detection-accuracy result.

| Pass | Frames | Full-loop FPS | Mean processing | p95 processing | p95 sensor-to-result age |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pixel A | 26 | 2.56 | 300.79 ms | 307.84 ms | 365.06 ms |
| Context A | 34 | 3.32 | 232.84 ms | 315.94 ms | 361.76 ms |
| Context B | 33 | 3.29 | 240.72 ms | 333.89 ms | 381.52 ms |
| Pixel B | 27 | 2.63 | 297.17 ms | 305.43 ms | 355.62 ms |

Combined full-loop rates: **2.59 FPS pixel, 3.31 FPS context**, about 27.6% higher
throughput for context in these configurations. This is not an equal-resolution
algorithm speedup: pixel searched 640×480 and context searched 320×240. Context's
p95 sensor age was not consistently better. Camera FPS requested at 10 does not
make either full processing loop deliver 10 FPS.

All recorded sensor timestamps increased. Raw/annotated first and last images
for each pass and every frame's timing/metadata are saved under
`runs/competition-comparison-20261006/pi-results/camera/`.

## What to do next for an actual competition comparison

1. Stage yellow and orange goals plus balloons in the Pi camera's view; record
   slow turns, approaches, partial occlusions, loss/return, and distractors.
2. Label those recordings and hold out whole sessions. The current public
   development images have already been used repeatedly for selection.
3. Evaluate the pixel detector on real 640×480 source recordings and calibrate
   its color settings on a separate calibration recording, not the test clips.
4. Compare mission-level target selection, including wrong-color rejection and
   goals only when possession is confirmed. This benchmark does not implement
   the mission state machine, physical distance, or capture confirmation.
5. Add the intended ESP32 communication load and a longer endurance run before
   calling these end-to-end competition rates. No motors are needed for these
   perception/transport bench tests.

## What is actually being tested

- Original Raspberry Pi Zero W / ARMv6, one processing method at a time except the alternating static-frame comparison.
- Updated pixel source: `/home/pacman/Documents/TESTING/hollow_test_again.py`, SHA-256 `815c5704151623196972466f8b241308f84714e742a1212318b4740df04a0ff5`.
- Context goal student: 64×64 FP32 crop CNN, SHA-256 `147671bbff83a8d0f57ff01a0fc8836441790b7b5c919f0e1ade1bb16c75fea3`.
- Rebuilt TensorFlow Lite C library: SHA-256 `54ed7fdda0b5b5f6db05a538eb800b7bc7ecd70d32f67dc13d1e78817d8b653a`.
- Pixel code imported without running its camera/web-server main. No changes to target colors, thresholds, geometry, or remembered-region behavior.
- No model retraining, motor commands, serial writes, network reconfiguration, package installation, or system-runtime replacement.

The updated pixel implementation searches for a hollow yellow region, remembers its box, and adjusts sampling stride. It selects the first accepted goal and does not identify circle/square/triangle or team-colored balloons. Its configured RGB target is `(156,156,0)`, color threshold 90, minimum shape size 20, base stride 1, memory enabled, adaptive stride enabled, ROI expansion 0.75. NumPy is the actual Pi backend.

The context path still searches/classifies both goal colors before filtering the
shared yellow-localization output. The timing includes that extra capability;
this is not a claim that the two methods perform identical internal work. Neither
comparison exercises balloon recognition or the capture/delivery state machine.

Its input is **BGR**, whereas our neural pipeline expects RGB. The comparison explicitly converts between them. A local synthetic wiring check found a hollow yellow square and rejected a solid yellow square, orange hollow square, and empty image. This is a software check, not an accuracy benchmark.

## Protocol

### 1. Common-input static accuracy

All 595 frozen development images at 320×240; original annotations and image hashes retained. Both methods receive the same pixels. Method order alternates. Pixel memory resets between unrelated photographs; the context path is stateless with twelve candidates. Common task: yellow-goal localization, ignoring shape, one-to-one IoU ≥0.5 matching.

Report all-target precision/recall and the fraction of target-containing frames in which at least one correct target is found. The latter helps interpret a method designed to choose one goal rather than enumerate every goal. It still does not prove the chosen goal is cheapest to reach.

The pixel method's native camera configuration is 640×480. This first comparison intentionally uses equal 320×240 inputs and leaves its pixel thresholds unchanged. It is not its optimized/native-resolution accuracy qualification.

### 2. Wall-clock-paced motion stress

Twelve deterministically selected development stills, two per goal class, each transformed into a three-second 10 FPS timeline. Transformations include gradual 15% scale increase, small translation, mild brightness reduction, a 300 ms black interruption, and return. Two repeats alternate method order.

Frames are pre-generated. Each method runs independently against actual elapsed wall time, takes the latest available frame, and skips missed frames instead of processing a backlog. Pixel ROI memory is enabled. Context uses four-crop temporal scheduling, native tracking differences, and unchanged freshness limits. State resets between clips, not between successive frames.

Report delivered FPS, processing p95, source-frame-to-result age, skipped frames, and detections during complete blackouts. Different methods process different temporal samples: their processed-frame accuracy figures are not a directly paired accuracy comparison. The static test above supplies the common-input comparison.

This is **generated stress replay**, not recorded competition motion. Blackouts are easier than real occlusion; transformed existing labels do not cover newly simulated physical effects. No physical speed, range, camera noise, rolling shutter, or flight geometry is established.

### 3. Native-configuration live-camera throughput

Bounded pixel/context/context/pixel passes, with a 640×480 camera stream requested at 10 FPS and no retained frame queue. Pixel searches 640×480; context resizes to its existing 320×240 search path. These are native processing configurations, not equal search resolution.

Capture, conversion, resizing, processing, and observation assembly contribute to delivered throughput. Report actual sensor timestamps and sensor-to-result age using the Pi boot clock. PNG/JSON disk writes occur after timed passes. No browser rendering or ESP32 transport is included. Short live runs include initial acquisition and are not a thermal endurance test.

The camera scene has no reviewed labels. Separate passes are not the same scene and cannot establish comparative detection accuracy. Saved raw and annotated images are for inspection, not automatic ground truth.

## Reproducibility

Local evidence: `runs/competition-comparison-20261006/`. Isolated Pi runner directory: `/home/pacman/competition-comparison-20261006/`.

Runners: [competition_compare.py](../scripts/competition_compare.py) and [competition_camera.py](../scripts/competition_camera.py). They import the frozen runtime package from `/home/pacman/pi-native-tracking-20261006` and select the rebuilt library per process with `DTR_TFLITE_LIBRARY`.

Full local regression suite: 235 passed, two skipped. New tests cover skipped-frame scheduling, the zero-transform identity, blackout labels, and return. Existing matching tests cover duplicate and one-to-one assignment behavior. Ruff passed.

The first paced attempt stopped on a JSON serialization error for transformed
NumPy coordinates. Coordinates were converted to native floats, a serialization
regression check added, and the experiment restarted into `paced-v2` artifacts.
The partial first-attempt file is retained and is not a completed result.

Final Pi temperature: 43.3°C; throttling flags `0x0`; 285 MiB available memory.
The existing swap occupancy (88 MiB) is not a measurement of benchmark paging.
The updated pixel file, original `demo.py`, and installed TFLite library retained
their pre-test SHA-256 values. The candidate runtime remains per-process only.
All 27 downloaded evidence/runner/image files matched their remote SHA-256 values.

Neither high crop accuracy nor high processing FPS qualifies autonomous capture, distance, goal entry, or flight. Competition qualification still needs labeled held-out recordings from the actual camera and arena.
