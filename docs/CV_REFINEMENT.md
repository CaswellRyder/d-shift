# Pi refinement experiments — 2026-10-06

Goal: reduce false detections and processing cost on the original ARMv6 Pi Zero W
without silently trading away detection quality. Existing viewer models, original
Pi demo, frozen comparison inputs and default HSV proposal policy are unchanged.
No motor commands, camera capture, network changes or package installs.

## Real training negatives, not generated accuracy claims

`scripts/mine_student_negatives.py` mined false-positive proposals from 2,950
training frames, producing 476 candidates and a 96-image review queue. Padded
crops had to avoid every annotated goal, including a guard margin. Visual review
still found real goals missing from annotations: annotation absence alone is not
a reliable background label.

All four contact sheets were visually reviewed by the assistant, not a human
label auditor. Fifty clear background crops were admitted through the explicit,
hash-bound `configs/student-negative-review-20261006.json`; ambiguous crops were
excluded. Several are correlated views, not 50 independent scenes. Original
labels and validation/test partitions were not edited. No synthetic training
images were added. This targeted real-error experiment was prioritized over image
generation; any future generated examples should remain training-only.

`scripts/refine_pi_student.py` starts from the warm separable student. Both
control and intervention run eight epochs with Adam 1e-4, seed 42, batch 32.
Original crops retain cached teacher targets (hard/KL weight 0.5, temperature 4).
New negatives receive hard background labels only, never fabricated teacher
logits. Fifty crops repeated eight times add 400 training entries. The control
has no added entries: epochs match, but optimizer updates do not (247 versus 260
batches per epoch). Validation hard-label loss selects the checkpoints.

### Same 595-frame development evaluation

320x240 inputs, 12 proposals, threshold 0.8, nested suppression, same-class
IoU >=0.5. This repeatedly used development set is not an independent test set.

| FP32 model | TP | FP | FN | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original context | 692 | 309 | 496 | 69.13% | 58.25% | 63.23% |
| Warm separable | 686 | 311 | 502 | 68.81% | 57.74% | 62.79% |
| Eight-epoch control | 685 | 353 | 503 | 65.99% | 57.66% | 61.55% |
| Added negatives | 684 | 321 | 504 | 68.06% | 57.58% | 62.38% |

Orange-square false positives decrease from 150 to 95 versus warm separable
(36.7%), but true positives fall from 50 to 48. Other classes regress: orange-circle
FP rises 30→59 and yellow-square FP 42→64. Versus the control, the negatives remove
23 orange-square FP at the same 48 TP. This is useful evidence for focused mining,
**not an overall improved model**. Neither checkpoint replaces the baseline.

Candidate coverage is unchanged: 735/1,188 goals, including only 15/284 tiny goals.
Retraining a crop classifier cannot recover objects never proposed to it.

Actual Pi golden-crop replay: 21 reference crops x three rounds, five warmups,
rotating model order, one thread. Warm FP32 averaged 43.16 ms/crop; negative-refined
FP32 43.32 ms/crop. All labels matched Mac references, maximum score error 1.64e-6.
Both have 8,279 parameters and 37,268-byte FP32 files. This experiment changes
weights, not inference complexity; it does not deliver a speed improvement.

## Exact RGB lookup — rejected as a default

An explicit `goal_lut` proposal profile uses a read-only 16 MiB RGB lookup instead
of repeated HSV conversion and thresholding. `scripts/build_goal_lookup.py`
builds it from the existing HSV policy. All 16,777,216 RGB values were verified
against both Mac OpenCV 4.11 and Pi OpenCV 4.10. Full proposal lists matched
exactly on all 595 development frames on both machines.

Paired Pi search-only timing alternated execution order, excluding decoding,
classification and initial table loading:

| Proposal implementation | Mean | p95 |
| --- | ---: | ---: |
| Existing HSV | 112.01 ms | 139.83 ms |
| RGB lookup | 120.72 ms | 149.84 ms |

The lookup was **7.77% slower**, despite exact outputs, and consumes extra memory.
Default remains HSV. The optional experiment requires an explicit
`DTR_GOAL_COLOR_LOOKUP` path and checksum/contract validation. Combined verifier,
crop and proposal benchmark process peak RSS was 141,756 KiB; this is not a
measurement of incremental lookup memory.

## Native tracking-difference implementation

`TemporalVision(..., difference_backend="native")` replaces repeated NumPy
absolute-difference/mean operations with `cv2.norm(..., NORM_L1)` and an equivalent
sum threshold. Internal signatures are integer-valued float32 arrays, and their
sums are exact at these sizes. Change thresholds, label lifetime, classification
budget and weights are unchanged. Unlike the lookup, this needs no extra table.
The original implementation remains selectable as `difference_backend="numpy"`.
Native is now the local source/benchmark CLI default. Frozen Pi bundles retain
their original source hashes; use `--difference-backend native` with the isolated
Pi `pi_temporal_bench.py` to select it explicitly. No original Pi demo is replaced.

Unit tests cover random signatures, both signature sizes and values immediately
below/at/above both decision thresholds. On the Mac, 240 replay frames with a
fixed processing clock produced identical complete outputs. The same 240-frame
audit on the Pi also completed with **zero output mismatches**. A fixed clock removes
processing-dependent label expiry solely for equivalence testing; its timings
must not be reported as real-time throughput.

### Actual Pi paired replay timing

Two runs, 240 frames per variant per run, fixed context FP32 weights:

| Run | NumPy mean / FPS | Native mean / FPS | NumPy / native p95 |
| --- | --- | --- | --- |
| First | 191.78 ms / 5.21 | 185.11 ms / 5.40 | 402.92 / 417.63 ms |
| Repeat | 193.79 ms / 5.16 | 185.26 ms / 5.40 | 430.57 / 416.55 ms |

Native processing throughput improves **3.60% and 4.60%**, respectively. These
are two paired runs, not a confidence-bounded speed guarantee. Tail latency does
not consistently improve. All four results have identical per-class counts and
63.48% replay F1, 439 neural calls and 78 scans. This does not change the static
model's accuracy. More than half the frames still exceed 100 ms; it is not a
10 Hz pipeline. The faster separable model was not used for this ablation.

### Re-run on the Pi without changing the original demo

From the isolated `/home/pacman/pi-native-tracking-20261006` bundle:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 benchmark_tracking_backends.py \
  --base ../pi-comparison-20261005 --model models/context.tflite \
  --output my-paired-replay.json
```

Use a fresh output name: the runner refuses overwrite. Add `--fixed-clock` only
for exact-output audits. Normal mode alternates variant order each frame, retains
real processing-dependent label expiry, and uses the same context weights and
four-crop budget. Both variants reset before each clip and before its timed work.
The workload is 12 development stills translated into 20 frames each, including
missing/reappearing targets. It excludes camera, decoding, frame generation,
rendering, transmission and result-file writes. It is not real flight footage.

## Reproducibility and evidence

- Mining queue and review: `data/goal-hard-negatives-20261006/` and
  `configs/student-negative-review-20261006.json`.
- Training: `runs/goal-refine-control-20261006/` and
  `runs/goal-refine-negatives-20261006/` (provenance, checkpoints and exports).
- Four-model development evaluation: `runs/cv-refinement-20261006/report.json`.
- Mac lookup: `runs/goal-lookup-20261006-mac.json`.
- Pi results: `runs/pi-refinement-results-20261006/`.
- Transfer bundles: `runs/pi-refinement-20261006/` and
  `runs/pi-native-tracking-20261006/`; source and model hashes are verified on Pi.

All eight downloaded Pi result files matched their remote SHA-256 values.
After the experiments, all 640 original comparison-bundle files still matched
their checksums. The original `Documents/TESTING/demo.py` SHA-256 remained
`f46b1dfde183e4695a9daa00ff8d3e9ca01a5ab1c85e969b9992b332a58ce8a7`.
Pi temperature was 42.2 C at completion; `get_throttled` reported `0x0`.
Available memory was 240 MiB, with 34 MiB swap occupied; swap occupancy alone
does not establish whether timed frames incurred swap I/O. SSH was closed.

No experiment here establishes competition accuracy, live-camera throughput,
capture confirmation, distance calibration, ESP32 integration or flight readiness.

## Next highest-value work

1. Improve small/faded-goal candidate coverage with bounded high-detail region
   searches, measuring proposal recall before spending time on more classifier
   training. Prior full-frame edge fallback regressed; do not enable it blindly.
2. Record representative Pi-camera sequences for blur, exposure shifts, distant
   orange goals and occlusion; retain independent recording sessions for evaluation.
3. Add diverse, reviewed background examples rather than repeatedly oversampling
   a small correlated set. Generated scenes may supplement training but should not
   serve as evidence of real detection accuracy.

Verification this round: 231 Python tests passed, two skipped; seven existing
viewer tests passed; Ruff passed. No existing viewer service was restarted.
