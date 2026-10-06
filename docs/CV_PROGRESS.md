# CV progress: uncertainty, smaller context model, actual Pi timing

Experiment started 2026-10-05. Hardware/capture-mechanism questions are not required
for this work. No motor commands, network changes, package installs, or startup
services. Original Pi demo, frozen comparison, and viewer model defaults remain intact.

## Implemented

- Viewer separates shape and color evidence. Strong shape / unknown color is
  visible as a purple tentative box, never silently accepted or called orange.
  This uses existing scores, not another inference. Suppression, cache expiry,
  and accepted-label policy are unchanged. See [uncertainty](GOAL_UNCERTAINTY.md).
- Per-frame timing now separates candidate search, crop classification, and
  observation assembly/suppression. This is processing time, not camera latency.
- Added a smaller `separable_context` research student: same 64x64 RGB input,
  three-layer stem, 2x2 layout pooling and seven-class head; replaces the final
  dense 3x3 context convolution with depthwise + pointwise convolutions.
- Added an optional trained-weight initializer: per-input-channel rank-one SVD
  factors the context kernel, while copying stem/head/bias, then normal
  distillation fine-tunes all weights. Source checkpoints are never modified.
- Added an opt-in grayscale edge candidate experiment within the same 12-crop
  budget. It regressed and remains disabled by default.

## Accuracy: development evidence, not competition qualification

Same 595 frozen development frames, 320x240, 12 proposals, threshold 0.8, nested
duplicate suppression, same-class matching at IoU >=0.5. These frames have been
used repeatedly for development; the reserved test split was not opened.

| FP32 model / candidate search | Parameters | Precision | Recall | Full-frame F1 |
| --- | ---: | ---: | ---: | ---: |
| Existing context / baseline | 16,183 | 69.13% | 58.25% | 63.23% |
| Separable, from scratch / baseline | 8,279 | 59.98% | 56.14% | 58.00% |
| Separable, trained-weight initialization / baseline | 8,279 | 68.81% | 57.74% | 62.79% |
| Existing context / edge fallback | 16,183 | 63.83% | 47.39% | 54.40% |

The warm-start model is **not a drop-in accuracy improvement**. Orange-square
precision falls from 62.67% to 25.00% (28 -> 150 false positives), and yellow-square
precision from 94.86% to 85.05%. Orange-circle and orange-triangle precision improve,
but aggregate F1 alone hides the square regressions. Keep the original context
as the accuracy reference and the warm-start model as an opt-in speed experiment.

The edge fallback replaces some color candidates to stay within budget. Candidate
coverage falls from 735/1,188 to 611/1,188 annotated goals, including only 8/284 tiny
goals instead of 15/284. On the Mac, search averaged 2.64 ms instead of 0.68 ms in
that paired run. **Rejected as a default.** Finding an artificial gray square in a
unit test does not establish useful proposal recall in actual scenes.

## Training details

Both new students use the same reviewed real crop manifest, frozen MobileNetV4
teacher logits, seed 42, temperature 4, equal hard-label/KL loss weighting,
Adam 1e-3, and 25 epochs. Checkpoints are selected by validation hard-label loss.
No added augmentation with stale cached targets. Scratch selected epoch 25;
warm-start selected epoch 22. The latter inherits a trained 25-epoch context model
and receives 25 additional epochs: this is **not an equal-training-budget test**.

SVD alone had 77.50% relative kernel approximation error and only 49.97% crop
accuracy. Fine-tuning is essential; kernel factorization is not output equivalence.
Final FP32 crop accuracies: existing context 98.31%, scratch separable 96.22%,
warm-start separable 97.97%. Warm-start INT8 crop accuracy is 97.86%.

Separable parameters decrease 48.84%. Theoretical convolution/dense MACs decrease
from 1,401,728 to 895,872 (36.09%); this is not a timing measurement. FP32 file sizes
are 68,064 vs 37,268 bytes; INT8 22,512 vs 16,280 bytes. Export formats are compared
on the actual Pi rather than assuming smaller files mean faster inference.

## Actual Pi Zero W crop benchmark

Original ARMv6 board, existing system TFLite C API, single-thread inference,
21 predecoded reference crops x three rounds = 63 timed calls per model; five
warmups each. Model order rotates/reverses to reduce fixed-order bias. Timing
includes resize, inference, softmax and shape/color evidence, not camera/search.

| Model | Format | Mean ms/crop | p95 ms | Reference parity |
| --- | --- | ---: | ---: | --- |
| Context | FP32 | 64.50 | 71.45 | Pass |
| Context | INT8 | 87.66 | 93.49 | Pass |
| Scratch separable | FP32 | 43.84 | 46.04 | Pass |
| Scratch separable | INT8 | 60.26 | 65.72 | Pass |
| Warm-start separable | FP32 | 43.65 | 47.14 | Pass |
| Warm-start separable | INT8 | 61.11 | 67.30 | Pass |

Warm-start FP32 has **32.32% lower mean crop latency** than context FP32 in this
run. Every label matched the Mac reference. Maximum FP32 score error was 1.64e-6;
maximum INT8 score error 0.00522, below the declared 0.03 tolerance. This is only
21 crop identities, not a complete cross-device accuracy test. FP32 is preferred
for further timing trials on this runtime; smaller INT8 files were slower.

The native library prints a CPU-topology parsing warning for package ID `-1`;
inference still completed, and parity passed. No package or runtime was replaced.

## Actual Pi controlled tracking replay

Completed 2026-10-06. Same deterministic 12 development stills, each translated
into 20 frames including two black missing-target frames: **240 generated frames**.
Both use the unchanged four-crop temporal policy, 0.5 s scan interval, 0.6 s class
refresh, 1.8 s high-score-background refresh, and 1.0 s label lifetime. The replay
clock advances a nominal 100 ms per frame, irrespective of actual processing time.

| FP32 model | Mean processing | p95 | Processing FPS | Six-class F1 | Yellow localization P / R |
| --- | ---: | ---: | ---: | ---: | ---: |
| Context | 193.43 ms | 432.11 ms | 5.17 | 63.48% | 96.05% / 72.65% |
| Warm-start separable | 159.02 ms | 351.51 ms | 6.29 | 59.34% | 88.24% / 70.51% |

Measured processing throughput increases **21.64%**, with 17.79% lower mean latency.
This is a single sequential run per model, not confidence-bounded performance.
It excludes camera, decoding, rendering, transport and file writes. These are
**not live-camera FPS** or real-motion accuracy, and neither trial achieves 10 Hz:
149/240 context and 143/240 warm-start calls exceed 100 ms.

Context made 439 neural calls and 78 full scans; warm-start made 452 and 77.
Their outputs affect tracking/scheduling, so they do not have identical inference
workloads. Classification ages reached 946.17 and 952.00 ms respectively. Both
had zero detections on the 24 black frames; both found at least one correct class
on the first returned frame in 11/12 clips, not proof of reacquiring every target.
The smaller model's replay F1 also regresses: keep both labeled experimental,
and do not replace the accuracy baseline merely for the higher FPS.

Native result JSONs and per-frame records are downloaded to
`runs/pi-cv-results-20261005/`. Replay runners verify the bundled source/model
hashes, original scoring helper/input-list hashes, and all 595 frame hashes before
running. The camera was not opened in these trials. The web viewer still uses
its original teachers, not either of these students.

After the runs, all **640 original comparison-bundle files** still matched their
checksums, and the original `Documents/TESTING/demo.py` SHA-256 remained
`f46b1dfde183e4695a9daa00ff8d3e9ca01a5ab1c85e969b9992b332a58ce8a7`.
Temperature was 34.7 C before and 41.2 C afterward; `get_throttled` reported `0x0`
at both checks. No flight process or camera stream was started. SSH was closed.

## Reproduce

Every output directory/file must be new; existing evidence is never overwritten.

```bash
PYTHONPATH=src TF_CPP_MIN_LOG_LEVEL=3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
.venv/bin/python scripts/distill_pi_student.py \
  --config configs/goal-proposals.json \
  --manifest data/dtr-proposals-reviewed-20261005/goal/manifest.json \
  --teacher runs/goal-proposals-20261005/teacher.keras \
  --target-cache-run runs/goal-pi-student-long-20261005 \
  --student-variant separable_context --alpha 0.5 --epochs 25 \
  --initialize-context runs/student-research-20261005/context-kd \
  --output runs/goal-separable-warm-repeat

PYTHONPATH=src .venv/bin/python scripts/export_pi_float.py \
  --run runs/goal-separable-warm-repeat \
  --output runs/goal-separable-warm-repeat/student.float.tflite

PYTHONPATH=src .venv/bin/python scripts/evaluate_cv_progress.py \
  --model context=runs/student-research-20261005/context-kd/student.float.tflite \
  --model warm=runs/goal-separable-warm-repeat/student.float.tflite \
  --profile balloon_components --output runs/cv-warm-repeat
```

Evidence locations:

- `runs/goal-separable-context-20261005/`: scratch training, metadata, FP32/INT8 exports.
- `runs/goal-separable-warm-20261005/`: initialization, inherited checkpoint hash,
  teacher/cache/manifest hashes, epoch log, crop validation and both exports.
- `runs/cv-progress-20261005/`: four-way model/search ablation, per-frame JSONL,
  stage timings and selected source snapshots.
- `runs/cv-warm-progress-20261005/`: matched context/warm full-frame results.
- `runs/pi-cv-progress-20261005/`: 47 hash-bound files, six model exports, 21
  reference crops and isolated native crop/replay runners. The same new directory
  is on the Pi under `/home/pacman/`; old benchmark directories remain untouched.

Native repeat commands (new output names required):

```bash
cd /home/pacman/pi-cv-progress-20261005
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 pi_research_crops.py \
  --bundle . --rounds 3 --output crops-repeat.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 pi_temporal_bench.py \
  --base ../pi-comparison-20261005 --model models/warm-fp32.tflite \
  --budget 4 --policy temporal --output warm-replay-repeat.json
```

Local software verification: 222 pytest tests passed, two skipped; Ruff passed;
seven Node tests passed. Added checks cover architecture/serialization, preserving
the original model, exact reconstruction of representable rank-one weights,
nonmutating SVD initialization, bounded edge proposals, stage-timing sums,
static-asset routing, and tentative display without changing acceptance.

Next accuracy work should target **train-only orange-square hard negatives and
small-goal proposal coverage**, not more indiscriminate architecture trials.
Real Pi-camera recordings with held-out sessions are still needed to establish
range, lighting/blur robustness and flight-scene accuracy. None of these models
estimates verified metric distance, confirms capture, or commands the ESP32.
