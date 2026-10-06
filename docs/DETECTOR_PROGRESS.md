# Learned goal localization — resumed experiment, 2026-10-05

**Current runtime status:** the original baseline completed at epoch 21 by its unchanged
patience-eight early-stopping rule. All baseline processes are terminal. No baseline model
passes the full standard; epoch 17 remains strongest at the fixed operating point.
A separate [reviewed-label refinement](DETECTOR_REFINEMENT.md) completed its 12-epoch cap from epoch 17.
Refinement epoch 2 is now strongest by weakest-metric ranking, but orange-triangle precision
(86.88%) and yellow-square recall (89.41%) still fail. Epochs 3–12 and finalized native best rank lower. Its earlier epoch-1
yellow pass did not persist; there is no full acceptance pass or combined-checkpoint result.
The baseline artifacts, validation labels, acceptance thresholds and viewer remain unchanged.
No trainer remains active for those completed runs. The [error/coverage review](GOAL_ERROR_FOLLOWUP.md)
led to a separate [four-epoch close-range exposure experiment](DETECTOR_CLOSE_EXPOSURE.md),
now running. Completed runs are not extended and no checkpoint is promoted.

## Completed baseline and next experiment

`comparison-completed-baseline.json` ranks all 21 saved epochs plus the finalized native
best checkpoint. Epoch 17 remains first; the finalized native best gives the same metrics
as epoch 13, so best mAP is not best fixed-confidence per-class acceptance.
Epoch 20 and 21 both fail. Final epoch 21 (`interim-u/development-640.json`):

| Class | Precision | Recall |
| --- | ---: | ---: |
| Orange circle | 95.12% | 93.60% |
| Orange square | 89.50% | 96.08% |
| Orange triangle | 74.90% | 95.02% |
| Yellow circle | 79.69% | 96.23% |
| Yellow square | 78.10% | 74.12% |
| Yellow triangle | 92.91% | 99.16% |

The finalized `last.pt` snapshot is bound to its embedded training-results epoch and the
completed progress epoch; stripping optimizer state does not erase this development evidence.
Snapshot capture still refuses failed/arbitrarily interrupted runs and never grants promotion.

The next run uses reviewed training labels, an epoch-17 EMA initialization, a fresh AdamW
optimizer, learning rate 0.0001, no warmup or mosaic, and a maximum of 12 epochs. This is a
compound refinement, not a controlled estimate of label-correction impact. All 3,545 exported
train/validation images and label hashes were rechecked before launch; validation is unchanged.
The original run is not extended or resumed with different data. Test remains untouched.

Verification after refinement and finalized-snapshot support: **135 tests pass**, two NMS
tests skip in the main environment and pass separately in the detector environment. Ruff
passes. Six refinement tests also pass directly in the detector environment. Existing four
TensorFlow Lite warnings remain; this is engineering proof, not an accuracy pass.

## Native-memory containment

While training epoch 14, free space fell from 24 to 19 GiB despite only 327 MiB in this run.
`vmmap -summary 80021` reported a **36.3-GiB physical footprint**, including 32.5 GiB in
Malloc Small regions, while the trainer GPU log showed only about 3.3 GiB. System swap used
15.7 GiB. This confirms process memory pressure; the exact native allocation cause is not
established. Do not infer that the earlier disk incident was entirely explained by this check.

SIGINT stopped only our trainer (exit 130). Saved epoch 13 remained loadable with optimizer
and EMA, SHA256 `d9e6e91047711d8786eaee1b5b3c5b3c0f1f228733cb75fd66144de4165be3d4`.
Partial epoch 14 is replayed. No files or other processes were removed/stopped.

`train_goal_detector.py` now defaults to **one completed epoch per process**. At the saved
epoch boundary it writes checksum-bound early-stopping state and exits **75**, with experiment
status `interrupted`, reason `process_epoch_limit`; this is an expected resumable boundary,
not completion/failure. It exits before final evaluation can strip optimizer state. Resume in
a fresh process with the same command and `--resume --max-process-epochs 1`; do not overlap
writers. The active goal handles subsequent resumes; no separate automatic supervisor exists.
The snapshot script accepts these saved boundaries as explicitly unapproved interim inputs.

Patience is preserved across these restarts. Legacy state is accepted only when the saved
epoch's fitness equals its recorded best (true for epoch 13); otherwise migration refuses
to guess. Final epoch/real early stopping still runs normal finalization. Restarts are not
bitwise augmentation replay and may alter the learning trajectory; this is a resource-policy
change, not a claim of equivalence to uninterrupted training. The 10-GiB disk guard remains.
The first bounded process completed epoch 14, saved resumable state and exited 75. The next
process successfully resumed epoch 15, preserving best-fitness epoch 13 and its 0.70867 value.
An in-flight `vmmap` sample of the bounded process measured 11.6 GiB (not its final peak);
free space remained about 19 GiB through the boundary. Sustained memory containment across
many segments still requires observation; the underlying native-allocation issue is not fixed.

### Random streams across restarts

Inspection showed the loader constructs a fixed-seed generator on every process start.
New code captures Python, NumPy, Torch CPU/MPS and both loader-generator states at each saved
boundary. `resume-state.json` binds the random-state file by SHA256 to the checkpoint; restore
occurs after trainer setup and before training iteration. This preserves random streams, not
bitwise training equivalence (optimizer serialization and dataset/iterator caches may differ).
For older boundaries without RNG state, an explicit epoch-keyed seed avoids identical repeated
shuffles. Epoch 15 records `legacy_epoch_seed: 56` and successfully saved the first RNG sidecar.
Its checkpoint SHA256 is `bd392200064100fd4fe445bae48403f436bb535aa493da12831f16a34e19a2fa`;
RNG sidecar SHA256 is `8ab9e37ba5140506f1bd77c9def58712e4989358fe153e25bcb61fafd52da284`.
Epoch 16's live `experiment.json` confirms mode `restored` with that same sidecar hash, followed
by nonzero training losses. Best-fitness epoch 13 / value 0.70867 is preserved across the
boundary. Random-stream tests now include an actual Torch serialization/deserialization round
trip. Free space remained about 19 GiB through both completed bounded segments; this supports
containment so far, not a fixed native allocator or a proof of sustained resource use.

Verification: **124 tests pass**, two NMS tests skip in the main environment and previously
passed separately in the detector environment; all three new random-stream tests also pass
directly in the detector environment. Ruff passes. No data or acceptance-standard changes.

### Full-precision continuation state

Inspection of installed Ultralytics 8.3.203 `DetectionTrainer.save_model` confirmed its ordinary
checkpoint saves `model=None`, an FP16 EMA model, and FP16 optimizer state; `load_checkpoint`
loads the EMA as the training model. Repeating that resume every epoch does not preserve live
raw weights/optimizer precision. This is a concrete continuation difference, not proof that it
caused any particular accuracy regression.

`dtr.training_checkpoint` now captures the raw training model, separate EMA, optimizer,
scheduler and scaler without downcasting, with CPU-cloned tensors. The checksum-bound
`resume-training-epoch-N.pt` sidecar includes this state alongside random streams. Restore
copies into existing parameters (preserving optimizer parameter identities). Legacy sidecars
are explicitly recorded as `legacy_ema_model_fp16_optimizer`, not silently described as exact
continuation. Epoch 18 saved the new format; its own startup still used legacy epoch-17 state.
Epoch 19 successfully restored the new format before its first batch. Restoration now compares
the complete saved model/EMA/optimizer/scheduler/scaler trees against a fresh capture, including
tensor dtype and exact values, and refuses any mismatch. The live provenance records
`training_state_resume: full_precision_raw_model_and_optimizer` and
`training_state_restoration_verified: true`, followed by nonzero training losses.

Epoch-18 checkpoint SHA256: `4bba5ca18696f3eff0579c881611ff749f9ef745d988adaab2925ef6eab4453b`.
Full-state sidecar SHA256: `38a715e568088ec2af24bc788767f4e9c531c07c1116059c539fade3f945e171`
(42,143,038 bytes). Inspection verified 418 FP32 model tensors, 81 integer buffers, 510 FP32
optimizer moment tensors, distinct raw/EMA weights, and matching checkpoint/sidecar bindings.

Three tests pass in the detector environment, including a serialized small-model AdamW
continuation with identical next-update weights and FP32 optimizer moments, and rejection of
precision changes/missing state by the exact comparator. This does not prove bitwise full-detector equivalence:
pending accumulated gradients and dataset/iterator caches are not restored. Normal inference
checkpoints and fixed-point evaluations remain unchanged; no unverified accuracy benefit claimed.

## Checked checkpoint: epoch 19

`interim-s/development-640.json`, all 595 development frames, CPU, fixed confidence 0.25:

| Class | Precision | Recall |
| --- | ---: | ---: |
| Orange circle | 95.24% | 96.00% |
| Orange square | 83.54% | 97.06% |
| Orange triangle | 75.10% | 93.03% |
| Yellow circle | 76.12% | 96.23% |
| Yellow square | 94.03% | 74.12% |
| Yellow triangle | 98.33% | 99.16% |

**FAIL:** orange-square/triangle precision, yellow-circle precision and yellow-square recall.
`comparison-through-epoch19-bound.json` retains epoch 17 as the strongest whole checkpoint.
Full-precision restoration is an engineering correction, not an observed accuracy improvement.
The epoch-19/20 boundary again passed exact training-state verification and restored RNG state
from sidecar SHA256 `e3a874ef73e2124e05aacf414931dfb9a6daca555408a8d1f2ccdc41cd58191a`.
Checkpoint SHA256: `896e84f22c449531982bf8c0f385a72b14572978d9b64723eb8b708ab0cf6f7c`.
Best trainer fitness remains 0.70867 at epoch 13; preserve the existing patience-eight rule,
not an extension to chase a passing result. Disk space remains about 19 GiB; corrected-data
training is still separate and unstarted. No candidate promotion or test inference.

## Checked checkpoint: epoch 18

`interim-r/development-640.json`, all 595 development frames, CPU, fixed confidence 0.25:

| Class | Precision | Recall |
| --- | ---: | ---: |
| Orange circle | 96.39% | 96.00% |
| Orange square | 88.74% | 96.57% |
| Orange triangle | 83.03% | 90.05% |
| Yellow circle | 79.27% | 96.23% |
| Yellow square | 81.25% | 76.47% |
| Yellow triangle | 98.33% | 99.16% |

**FAIL:** orange-square/triangle precision, yellow-circle precision, yellow-square precision
and recall. Epoch 17 remains the top whole-checkpoint candidate in
`comparison-through-epoch18-bound.json`; no inference configuration is promoted. Disk space
remained about 19 GiB. The full-precision continuation change begins with epoch 19 and cannot
be credited with improving epoch 18 (which is below the acceptance standard).

## Checked checkpoint: epoch 17

`interim-q/development-640.json`, all 595 development frames, CPU, fixed confidence 0.25:

| Class | Precision | Recall |
| --- | ---: | ---: |
| Orange circle | 94.49% | 96.00% |
| Orange square | 91.24% | 97.06% |
| Orange triangle | 86.55% | 96.02% |
| Yellow circle | 98.71% | 96.23% |
| Yellow square | 95.76% | 88.63% |
| Yellow triangle | 98.33% | 99.16% |

**FAIL, but strongest whole checkpoint so far:** orange-triangle precision and yellow-square
recall remain below 90%. Both color aggregates exceed 90% (orange 90.92% / 96.34%, yellow
97.26% / 93.25% precision/recall), but aggregates do not replace class requirements.
`comparison-through-epoch17-bound.json` ranks epoch 17 first: weakest required metric 86.55%,
up from epoch 11's 78.46%. No model promotion or threshold change.

`interim-q/errors-640.json` records 17/30 orange-triangle false positives with insufficient
same-class IoU, and 28/29 yellow-square misses with no prediction overlapping at the diagnostic
threshold; 27 yellow-square misses are in the >=32px stratum. These describe the supplied
labels, not new training labels. Reserved test, corrected-data export and viewer remain unchanged.

## Checked checkpoint: epoch 16

`interim-p/development-640.json`, all 595 development frames, CPU, fixed confidence 0.25:

| Class | Precision | Recall |
| --- | ---: | ---: |
| Orange circle | 93.81% | 84.80% |
| Orange square | 89.50% | 96.08% |
| Orange triangle | 94.27% | 90.05% |
| Yellow circle | 98.71% | 96.23% |
| Yellow square | 47.20% | 99.22% |
| Yellow triangle | 98.33% | 99.16% |

**FAIL:** orange-circle recall, orange-square precision, yellow-square precision. The orange
square's 89.50% must not be rounded into a pass. `comparison-through-epoch16-bound.json`
retains epoch 11 as the strongest whole-checkpoint candidate; no candidate is promoted.

`interim-p/errors-640.json`: 225/283 yellow-square false positives have no labeled overlap,
35 have insufficient same-class IoU, and 21 are duplicates at matching IoU. All 38 missed
orange circles lie in the under-16px reporting stratum. These are label-relative diagnostics,
not verified negative labels: any future hard-negative mining must use reviewed training
frames, not validation failures relabeled and added to training. Training remains unchanged.
The epoch-16/17 boundary again preserves random streams and early-stopping state; disk space
remained about 19 GiB. No test inference, distillation, Pi deployment or flight qualification.

## Checked checkpoint: epoch 15

`interim-o/development-640.json`, all 595 development frames, CPU, fixed confidence 0.25:

| Class | Precision | Recall |
| --- | ---: | ---: |
| Orange circle | 95.98% | 76.40% |
| Orange square | 71.27% | 96.08% |
| Orange triangle | 85.33% | 95.52% |
| Yellow circle | 78.46% | 96.23% |
| Yellow square | 47.83% | 81.96% |
| Yellow triangle | 95.16% | 99.16% |

**FAIL:** only yellow triangle meets both class thresholds. There are 228 yellow-square
false positives. `comparison-through-epoch15-bound.json` retains epoch 11 as the strongest
whole-checkpoint candidate, itself still failing the all-six-class standard. No candidate
promotion or threshold change. Keep the existing training plan/early stopping; corrected data
remains a separately prepared next experiment, not silently substituted during resume.

## Checked checkpoint: epoch 14

`interim-n/development-640.json`, complete 595-frame CPU evaluation, unchanged confidence 0.25:

| Class | Precision | Recall |
| --- | ---: | ---: |
| Orange circle | 90.17% | 84.40% |
| Orange square | 74.43% | 95.59% |
| Orange triangle | 94.53% | 94.53% |
| Yellow circle | 85.96% | 96.23% |
| Yellow square | 67.83% | 99.22% |
| Yellow triangle | 98.33% | 99.16% |

**FAIL:** orange-circle recall; orange-square, yellow-circle and yellow-square precision.
Only the two triangle classes meet both thresholds. `comparison-through-epoch14-bound.json`
still ranks epoch 11 first by the recorded whole-checkpoint rule. No model promotion,
viewer change, distillation, test inference or Pi deployment occurred.

## Checked checkpoint: epoch 13

`interim-m/development-640.json`, 595-frame CPU evaluation at the unchanged operating point:

| Class | Precision | Recall |
| --- | ---: | ---: |
| Orange circle | 96.73% | 82.80% |
| Orange square | 93.81% | 96.57% |
| Orange triangle | 83.11% | 90.55% |
| Yellow circle | 88.44% | 96.23% |
| Yellow square | 74.17% | 87.84% |
| Yellow triangle | 97.52% | 99.16% |

**FAIL:** orange-circle recall, orange-triangle precision, yellow-circle precision, and
yellow-square precision/recall. Higher trainer mAP50-95 does not imply improvement at the
fixed acceptance point. `comparison-through-epoch13-bound.json` retains epoch 11 as the
strongest whole-checkpoint candidate by the recorded rule; it too remains below the full
standard. Training continues without a dataset, threshold or runtime-default change.

## Checked checkpoint: epoch 12

`interim-l/development-640.json`, complete 595-frame CPU evaluation at confidence 0.25:

| Class | Precision | Recall |
| --- | ---: | ---: |
| Orange circle | 93.39% | 90.40% |
| Orange square | 53.49% | 97.55% |
| Orange triangle | 92.35% | 90.05% |
| Yellow circle | 98.08% | 96.23% |
| Yellow square | 43.87% | 89.80% |
| Yellow triangle | 98.33% | 99.16% |

**FAIL:** orange-square precision and yellow-square precision/recall. False positives rise
to 173 orange squares and 293 yellow squares. Other classes' passing results must not hide
this regression. `comparison-through-epoch12-bound.json` retains epoch 11 as the top whole
checkpoint by the recorded weakest-metric/macro-F1 rule. No promotion or threshold change.

An extra epoch-11 cross-class NMS diagnostic (`interim-k/cross-class-nms.json`, fixed IoU 0.7)
removes 30 yellow-circle false positives but also four yellow-square true positives. Yellow
square still fails and the policy remains **unselected**. This check supports the duplicate
shape hypothesis without treating suppression as a proven solution. The baseline remains
unchanged; wait for completion before starting the separately prepared corrected-data run.

## Checked checkpoint: epoch 11

`interim-k/development-640.json`: full 595-frame CPU evaluation, unchanged confidence 0.25,
same-class IoU >= 0.5. **All three orange classes meet both development targets in this
single raw-detector checkpoint. The complete six-class standard still fails on yellow.**

| Class | Precision | Recall | Provisional class target |
| --- | ---: | ---: | --- |
| Orange circle | 93.98% | 93.60% | Meets development target |
| Orange square | 94.23% | 96.08% | Meets development target |
| Orange triangle | 94.53% | 94.53% | Meets development target |
| Yellow circle | 78.46% | 96.23% | Fail: precision |
| Yellow square | 85.71% | 91.76% | Fail: precision |
| Yellow triangle | 98.33% | 99.16% | Meets development target |

Aggregate orange **94.22% precision / 94.66% recall**; yellow **85.88% / 94.75%**.
`comparison-through-epoch11-bound.json` ranks epoch 11 first: it ties epoch 8 on the weakest
required metric (78.46%) and wins the recorded macro-F1 tiebreak. Ranking is diagnostic,
not promotion, independent qualification, or evidence of 320-input/Pi performance.

`interim-k/errors-640.json` identifies the remaining error mix relative to supplied labels:
35/42 yellow-circle false positives overlap another labeled class at matching IoU;
22/39 yellow-square false positives overlap a same-class target below matching IoU.
This directs follow-up toward shape confusion and localization rather than assuming every
false positive is background. Labels and thresholds are unchanged; baseline training continues.

## Checked checkpoint: epoch 10 and training-label corrections

`interim-j/development-640.json` evaluates the complete 595-frame development set at the same
fixed confidence 0.25 on CPU. **The standard still fails; epoch 10 regresses on yellow.**

| Class | Precision | Recall |
| --- | ---: | ---: |
| Orange circle | 92.75% | 97.20% |
| Orange square | 85.28% | 96.57% |
| Orange triangle | 87.33% | 96.02% |
| Yellow circle | 57.68% | 96.86% |
| Yellow square | 88.99% | 79.22% |
| Yellow triangle | 98.33% | 99.16% |

Only orange circle and yellow triangle meet both thresholds at this checkpoint. Epoch 9's
four passing classes must not be attributed to epoch 10. Whole-checkpoint comparison remains
in `comparison-through-epoch10-bound.json`; no model promotion or runtime-default change.

Visual inspection of epoch-9 square-error contact sheets (`interim-i/review/`) found yellow
goals predicted as orange and a yellow floor object generating false squares, alongside apparent
annotation issues. A train-only color scan followed by inspection of both complete original
images confirmed two distinct training errors:

- Annotation 650: yellow square goal labeled orange square. Preserve the box, correct color.
- Annotation 4675: green balloon labeled orange square goal. Remove this false goal annotation;
  leave the separate balloon dataset unchanged.

`configs/goal-training-color-review-20261005.json` binds these AI visual decisions to the exact
source annotation hash, image hashes, annotation IDs, original labels and original boxes.
The exporter now supports explicit `relabel` / `remove_non_goal` decisions, train-only,
rejecting changed evidence or conflicts with duplicate-review decisions. It does not infer
corrections from color thresholds, create boxes, or modify source annotations.

New **untrained** export `data/goal-detector-v10-reviewed-color-20261005/` combines the prior
five duplicate corrections with these two decisions: **2,950 training frames / 2,826 goal
labels**. Its `comparison.json` verifies exactly two changed training label files relative to
the duplicate-reviewed export, identical validation, and every image/label hash in both exports.
Receipt SHA256: `f88cff3e4a0e1d1b75acf988bca62a57c35d7d8a519f1bdea056a062702ef282`.
The active baseline still uses its original dataset. Finish it before a separate corrected-data
experiment; do not resume with a changed receipt. Reserved test remains unread.

Verification: **118 tests pass**, 2 detector-specific tests skip in the Keras environment and
pass separately via `.detector-venv/bin/python tests/test_detector_nms.py`; Ruff passes.
These are engineering checks, not evidence that the corrected data improves accuracy.

## Post-recovery checkpoint: epoch 9

All 2,950 training and 595 validation exported images and label files passed their recorded
hash checks after storage cleanup. The reserved test was not read. `interim-i/` captures the
completed epoch-9 checkpoint immutably; `development-640.json` evaluates all 595 development
frames on CPU, at the unchanged confidence 0.25 and IoU >= 0.5 standard.

| Class | Precision | Recall | Provisional class target |
| --- | ---: | ---: | --- |
| Orange circle | 95.0% | 92.0% | Meets development target |
| Orange square | 77.3% | 96.6% | Fail: precision |
| Orange triangle | 94.1% | 95.5% | Meets development target |
| Yellow circle | 98.7% | 96.2% | Meets development target |
| Yellow square | 78.4% | 95.3% | Fail: precision |
| Yellow triangle | 98.3% | 99.2% | Meets development target |

Orange aggregate: **88.3% precision / 94.5% recall**; yellow: **87.9% / 96.4%**.
Four classes meet both thresholds in the same checkpoint, but the six-class standard fails.
`comparison-through-epoch9-bound.json` still ranks epoch 8 first by weakest required metric
(78.46%, versus epoch 9's 77.25%); more passing classes does not imply a better worst-class
result. No checkpoint was promoted.

`interim-i/errors-640.json` partitions the cached errors: 34 of 58 orange-square false
positives overlap a different labeled class at matching IoU; 56 of 67 yellow-square false
positives have no labeled overlap at the diagnostic threshold. These are descriptive
label-relative categories, not proof of background objects or authorization to change labels.
Training continues on the original dataset; reviewed training-label corrections remain a
separate future experiment. Viewer defaults, distillation, reserved test and Pi stay unchanged.

### Earlier storage interruption

The prior process terminated during epoch 9 with exit 137 after repeated
disk-full errors. The exact kill mechanism is unconfirmed. PID 4113 is gone and its exec session
was terminal, despite stale `running` metadata. That metadata was corrected to
`interrupted` before the new resume. Epoch 8's checkpoint was intact and loadable with optimizer/EMA state, byte-identical
to `interim-h/model.pt`. The experiment is incomplete and no detector passes the full standard.

At interruption about **5 GiB** was free on the training volume. Inspected MPS temporary directories were empty;
no useful cleanup space was found there. No personal files, datasets, checkpoints or caches were
deleted. A new guard checks both output and temporary volumes **before model import** and
during train/validation batches, requiring at least **10 GiB free**. A live resume preflight
correctly refused at 5.07 GiB without starting a trainer. This reserve reduces risk; it does not
guarantee against abrupt external space/memory pressure. Aim for 15–20 GiB free before resuming.

After freeing space and confirming no other writer is running:

```bash
PYTHONPATH=src .detector-venv/bin/python scripts/train_goal_detector.py \
  --data data/goal-detector-v10-20261005 \
  --output runs/goal-detector-safe-20261005 --epochs 30 --batch 8 --resume
```

This continues from saved epoch 8; partial epoch 9 is replayed. Keep the reviewed-label dataset
for a separately recorded experiment, not a silent substitution during resume. Earlier sections
below describe historical progress while the process was live.

## Objective and standard

Both neon orange and neon yellow goals are required. The original color-proposal + MobileNet
path is still below the target and remains the viewer's selected research configuration.
The new experiment learns full-frame goal boxes with a pretrained YOLO11n detector on the Mac.
It does not replace MobileNetV4, start distillation, or claim an ARMv6 deployment path.

`configs/goal-detection-standard.json` records a **provisional**, user-unconfirmed engineering
target: at least **90% precision and 90% recall for each of six color/shape classes**, same-class
one-to-one IoU>=0.5, confidence 0.25, at least 50 labeled targets per class. Color aggregates,
class-agnostic localization, and class/size strata must also be reported. Aggregate mAP alone
is not a pass. Independent data, actual neon-goal camera evidence and Pi runtime measurements
are separate qualification gates; detector metrics never approve autonomous flight.

## Data and environment

- Separate `.detector-venv`; existing Keras `.venv` and viewer remain unchanged.
- `requirements-detector.in` and resolved `requirements-detector.txt` pin the experiment.
- Ultralytics 8.3.203, PyTorch 2.6.0, YOLO11n COCO-pretrained initialization; official model
  URL/checksum and separate AGPL-3.0/Enterprise notice are in `THIRD_PARTY_NOTICES.md`.
- Derived dataset `data/goal-detector-v10-20261005/`: **2,950 train / 595 validation frames**,
  2,832 / 1,188 goal boxes. All six classes retained, including tiny positives previously
  excluded from some classification-crop datasets. No new synthetic or pseudo labels.
- Same grouped V10 split; excludes the previously reviewed conflicting training source.
  The entire reserved test set is **not read, exported or evaluated** by the converter.
- Images are hardlinked, labels are converted to normalized YOLO boxes. Receipts retain
  source annotation, image and label hashes. Missing upstream annotations and correlated
  recordings remain limitations; no full annotation audit is claimed.
- Local settings disable external logging/HUB/sync. No cloud training, uploads or paid compute.

## Training fault found and fixed

An initial MPS run had zero box and DFL losses on every training batch. It was stopped, not
interpreted as successful learning. A guarded reproduction also failed fast. Capturing the
first augmented training batch exposed corrupted label tensors after non-blocking device
transfers: **0/9 valid positive boxes**, with garbage/zero coordinates. CPU/MPS direct-loss
checks on retained input tensors agreed, narrowing the fault to the transfer path.

A local trainer/validator adapter uses **blocking tensor transfers**. The corrected captured
batch has **9/9 valid positive boxes**, and box/DFL losses are nonzero during real training.
This is a workaround verified for the installed environment, not an upstream bug-fix claim.
Training also rejects invalid box coordinates and aborts after ten consecutive labeled batches
with no box-regression signal. The validation path uses blocking transfers too.

Evidence retained:

- `runs/detector-loss-check-20261005.json`: CPU/MPS loss checks.
- `runs/detector-transfer-diagnostic-20261005.json`: captured-batch comparison and hashes.
- `runs/goal-detector-20261005/`: interrupted zero-regression run, not a candidate.
- `runs/goal-detector-guarded-20261005/`: automatically rejected reproduction, not a candidate.
- `runs/goal-detector-mpssafe-20261005/`: short training-only fix verification, interrupted
  before restarting with the validation transfer fix. Not a selected model.
- **`runs/goal-detector-safe-20261005/`**: current clean experiment. Inspect its live process,
  `progress.json`, `fit/results.csv` and `experiment.json`; a stale status file alone does not
  prove it is running or complete.

An additional validation fault appeared after epoch 2: Ultralytics' default NMS wall-time
cutoff emitted `NMS time limit 2.800s exceeded`. Its implementation breaks out of the batch
loop, leaving later frames with empty predictions. Those trainer metrics/rankings are not
trusted. Training was interrupted during epoch 3 and resumed from the saved epoch-2 state.
`before-validation-v2-*` preserves earlier checkpoints/metrics; `resume_history` records the
checkpoint hash. Prior best-fitness ranking was reset, without discarding learned weights,
optimizer state or EMA. Partial epoch 3 is replayed; augmentation replay is not bitwise exact.

Validation revision `blocking-transfer-complete-cpu-nms-v2` copies predictions to CPU, runs
NMS without the silent time cutoff, then returns results to the validation device with
blocking transfers. The fixed-confidence evaluator uses the same complete CPU NMS path.
The model outputs are cloned because NMS edits box coordinates in place. This is for
offline accuracy measurement, **not** a bounded-latency flight runtime. A regression test
reproduces skipped frames under a simulated clock jump, then verifies all 16 frames are
processed with the adapter; a second test checks CPU/MPS parity.

Current plan: up to 30 epochs, early-stopping patience 8, 640-pixel detector input, batch 8,
MPS, AdamW 0.001, frozen hue augmentation to preserve goal colors. Other augmentation and
exact parameters are retained in `fit/args.yaml`. This uses more input detail/compute than
the old 320x240 proposal path; comparisons must state that difference.

## Commands

Run from the project root. Do not start a second writer to the active output directory.
The training command refuses an existing destination unless explicit resume is requested;
resume is only appropriate after checking that the prior process has stopped and that an
epoch checkpoint exists. Resume is checkpoint-based, not bitwise replay of augmentations.

```bash
# Already prepared and running; shown for reproduction with FRESH output paths.
uv venv --python .venv/bin/python .detector-venv
uv pip sync --python .detector-venv/bin/python requirements-detector.txt
PYTHONPATH=src .detector-venv/bin/python scripts/train_goal_detector.py \
  --data data/goal-detector-v10-20261005 \
  --output runs/goal-detector-safe-20261005 --epochs 30 --batch 8

# Only after the selected run completes; requires its recorded model checksum.
PYTHONPATH=src .detector-venv/bin/python scripts/evaluate_goal_detector.py \
  --run runs/goal-detector-safe-20261005 --input-width 640 \
  --output runs/goal-detector-safe-20261005/development-640.json
PYTHONPATH=src .detector-venv/bin/python scripts/evaluate_goal_detector.py \
  --run runs/goal-detector-safe-20261005 --input-width 320 \
  --output runs/goal-detector-safe-20261005/development-320.json

# Optional interim diagnostics while training; use a fresh snapshot directory.
PYTHONPATH=src .detector-venv/bin/python scripts/snapshot_goal_detector.py \
  --run runs/goal-detector-safe-20261005 --output runs/goal-detector-safe-20261005/interim-new
PYTHONPATH=src .detector-venv/bin/python scripts/evaluate_goal_detector.py \
  --snapshot runs/goal-detector-safe-20261005/interim-new --input-width 640 --device cpu \
  --output runs/goal-detector-safe-20261005/interim-new/development-640.json
```

The 640 mode uses original source frames; the 320 mode first resizes to the viewer's 320x240
input. That changes aspect ratio as well as resolution relative to square training inputs.
Both match predictions in original annotation coordinates. The evaluator refuses unfinished
runs unless given an explicit immutable interim snapshot; all snapshots forbid promotion.
It checks checkpoint/standard/dataset-receipt hashes, validation annotation and image hashes,
frame membership, and class order. It never evaluates the reserved test split or approves deployment.

## Interim epoch-1 evidence — not final model qualification

`runs/goal-detector-safe-20261005/interim-a/` contains an immutable checkpoint and complete
CPU evaluations of all 595 validation frames. Confidence was fixed at 0.25, IoU at 0.5.
No time-limit warnings occurred in these single-frame CPU evaluations.

| Class | Precision, 640 | Recall, 640 | Provisional class target |
| --- | ---: | ---: | --- |
| Orange circle | 85.5% | 37.6% | Fail |
| Orange square | 57.9% | 66.7% | Fail |
| Orange triangle | 76.5% | 97.0% | Fail |
| Yellow circle | 91.1% | 96.9% | Meets development target |
| Yellow square | 88.5% | 72.2% | Fail |
| Yellow triangle | 92.9% | 99.2% | Meets development target |

Orange aggregate at 640: 70.8% precision / 64.9% recall. Yellow: 90.5% / 85.6%.
At 320x240, orange falls to 59.3% / 10.2%, yellow to 90.2% / 60.6%; **all six classes fail**.
This is not a deployable low-resolution detector or evidence of Pi performance.

`interim-a/review/` contains contact sheets and error receipts derived only from the cached
validation predictions. Visual review found orange circles called squares, localization errors
on tiny goals, duplicate boxes, and missing large yellow squares. These are diagnosis aids,
not verified label corrections. No validation images/labels were moved into training.
Keep the independent Highbay holdout sealed until model selection is finished.

`interim-b/` retains epoch 2 with complete-CPU-NMS evaluation on the same 595 frames:
orange **70.5% precision / 68.2% recall**, yellow **29.1% / 91.6%**. All six class targets
fail. The recall improvement is accompanied by many extra yellow detections and poor boxes;
this checkpoint is not an improvement suitable for promotion. Its `review/` contact sheets
retain the failures. Training is still in progress; do not present epoch 1's better yellow
precision as the current checkpoint's result or report trainer mAP as fixed-threshold precision.

## Checked checkpoint: epoch 3

The resumed run completed epoch 3 with full-batch CPU NMS and began epoch 4. The immutable
`interim-c/` checkpoint was independently evaluated on CPU across the same 595 validation
frames at 640 input / confidence 0.25. **Development standard still fails.**

| Class | Precision | Recall | Provisional class target |
| --- | ---: | ---: | --- |
| Orange circle | 96.1% | 48.8% | Fail: recall |
| Orange square | 63.4% | 95.1% | Fail: precision |
| Orange triangle | 80.2% | 86.6% | Fail: both |
| Yellow circle | 68.0% | 96.2% | Fail: precision |
| Yellow square | 93.8% | 82.7% | Fail: recall |
| Yellow triangle | 98.3% | 99.2% | Meets development target |

Orange aggregate: **75.4% precision / 74.8% recall**. Yellow: **84.6% / 90.4%**.
These are fixed-confidence development results, not the trainer's confidence-swept metrics.
Yellow false positives fell from 1,191 at epoch 2 to 88 at epoch 3. Training continues;
no model was promoted and the viewer, reserved test, distillation and Pi remain untouched.
The same checkpoint at 320x240 still fails every class: orange **55.9% precision / 18.0%
recall**, yellow **89.4% / 63.2%**. Higher-resolution progress must not be presented as a
solution to the Pi's low-resolution/compute constraints.

### Error diagnosis and rejected cross-class suppression

`scripts/diagnose_goal_detector.py` partitions cached errors at the unchanged operating point.
It validates inputs and conserves false-positive/missed-target counts. Epoch 3's
`interim-c/error-partition-640.json` shows 99 of 128 missed orange circles in the under-16px
equivalent size bin. Of 72 yellow-circle false positives, 71 overlap a different labeled class
at IoU >=0.5. This is descriptive overlap evidence, not an assertion that all upstream labels
are correct. The error contact sheets also contain apparent label mistakes; no labels changed.

An additional, class-agnostic NMS pass at the existing 0.7 overlap setting was tested on cached
epoch-3 predictions (`scripts/audit_detector_nms.py`, `interim-c/nms-ablation-640.json`).
It removed 99 predictions: 82 false positives **and 17 true positives**, including 14 yellow
squares. It still fails five class targets. **Rejected as default**; training, evaluator and
viewer retain their class-aware NMS. This is not a precision-only promotion.

## Checked checkpoint: epoch 4

`interim-d/development-640.json`: the full 595-frame CPU check at confidence 0.25 remains
**FAIL**. Orange triangles now meet the provisional target, but other classes regress.
Epoch-specific successes must not be combined into a fictitious single passing model.

| Class | Precision | Recall | Provisional class target |
| --- | ---: | ---: | --- |
| Orange circle | 95.2% | 55.6% | Fail: recall |
| Orange square | 80.7% | 55.4% | Fail: both |
| Orange triangle | 91.0% | 95.5% | Meets development target |
| Yellow circle | 57.4% | 95.6% | Fail: precision |
| Yellow square | 36.3% | 76.1% | Fail: both |
| Yellow triangle | 98.3% | 99.2% | Meets development target |

Color aggregates: orange **89.3% precision / 67.8% recall**, yellow **50.4% / 87.1%**.
No promotion. Continue the same bounded training experiment before selecting a checkpoint;
early-epoch metrics fluctuate, and the immutable earlier snapshots remain available.

## Checked checkpoint: epoch 5

`interim-e/development-640.json` uses the unchanged operating point and full 595-frame CPU
evaluation. Overall still **FAIL**, despite meaningful orange improvement.

| Class | Precision | Recall | Provisional class target |
| --- | ---: | ---: | --- |
| Orange circle | 92.40% | 92.40% | Meets development target |
| Orange square | 91.75% | 92.65% | Meets development target |
| Orange triangle | 89.77% | 96.02% | Fail: precision, do not round up |
| Yellow circle | 81.05% | 96.86% | Fail: precision |
| Yellow square | 44.53% | 89.41% | Fail: both |
| Yellow triangle | 98.33% | 99.16% | Meets development target |

Orange aggregate **91.36% / 93.59% precision/recall**; yellow **60.83% / 93.81%**.
At 320x240 all classes still fail: orange **67.08% / 24.89%**, yellow **78.09% / 68.86%**.
`error-partition-640.json` records the remaining fixed-threshold errors. Orange misses are
now concentrated in the smallest target bin; yellow squares still have many bad/extra boxes.

An audit of the **existing**, unchanged same-class nested suppression policy is retained in
`interim-e/nested-ablation-640.json` (reproduce with `scripts/audit_detector_nms.py --policy nested`).
It removes 85 false positives but also nine true positives. Orange circle/square/triangle
precision-recall become **93.85/91.60**, **92.65/92.65**, **91.00/95.52%**, respectively:
all three orange class thresholds are met in this development-only ablation. Yellow circle
and square still fail; six yellow-square true positives are lost. **Not selected or promoted**;
do not conflate this policy's numbers with the raw detector or independent/Pi qualification.

## Reviewed training labels for the next experiment

A train-only overlap scan found six pairs at IoU >=0.7. One source was already excluded.
Visual inspection of the remaining five original images confirmed three single yellow circles
also labeled as squares, one orange triangle with two boxes, and one partial yellow circle
with near-identical duplicate boxes. The checksum-bound decisions are in
`configs/goal-training-label-review-20261005.json`.

The exporter now optionally accepts `--training-review`. It verifies training-only scope,
annotation/image hashes, exact dropped/retained IDs, retained class and geometric overlap.
It removes only the reviewed erroneous/duplicate annotation, retaining the existing correct
box and all other labels. **No new boxes or validation-label corrections were introduced.**

Prepared separately: `data/goal-detector-v10-reviewed-20261005/`, 2,950 training frames and
2,827 goal labels (five fewer); validation remains byte-identical with 595 frames/1,188 goals.
`comparison.json` verifies all original exported label hashes and exactly five changed
training label files. Original annotations and the active training dataset are unchanged.
The reserved test was not read. The baseline trainer is still running on its original dataset;
the corrected export is **not yet used for training** and has no accuracy improvement claim.

```bash
PYTHONPATH=src .venv/bin/python scripts/prepare_goal_detector.py \
  --output data/goal-detector-v10-reviewed-20261005 \
  --training-review configs/goal-training-label-review-20261005.json
```

This is a targeted AI review, not a full human annotation audit. Missing edge targets and
other label mistakes may remain. Finish the current baseline before a separately recorded
corrected-data experiment; do not silently change labels under its running trainer.

## Checked checkpoint: epoch 6

`interim-f/development-640.json` retains the fixed-confidence CPU evaluation. All classes now
have at least 92% recall, but three fail precision. **Overall FAIL**; no checkpoint promotion.

| Class | Precision | Recall | Provisional class target |
| --- | ---: | ---: | --- |
| Orange circle | 77.26% | 92.40% | Fail: precision |
| Orange square | 57.80% | 98.04% | Fail: precision |
| Orange triangle | 90.32% | 97.51% | Meets development target |
| Yellow circle | 98.71% | 96.23% | Meets development target |
| Yellow square | 31.16% | 99.22% | Fail: precision |
| Yellow triangle | 98.33% | 99.16% | Meets development target |

A diagnostic confidence sweep over 0.25, 0.5, 0.75, 0.9 and 0.95 found no passing global
operating point. Raising confidence alone is not a solution: yellow-square precision remains
below 40% at every tested value and recall falls. The acceptance config remains unchanged.

### Rejected offboard MobileNet verification cascade

`scripts/verify_detector_candidates.py` evaluates cached detector boxes with the existing
proposal-adapted MobileNetV4 goal verifier. It uses the original 640-frame pixels, 12% crop
padding, the verifier's stored 0.8 acceptance threshold, and same-class NMS at 0.7 after
reclassification. The original detector score remains the ranking/0.25 gate; verifier scores
are a separate gate, not a rescaling that disguises a new operating point. No ground-truth
boxes are passed to the verifier, no models are trained, and inference runs on CPU.

`interim-f/verified-640.json` stores each crop decision, hashes and complete measured results.
Compared to raw epoch 6, yellow-square false positives fall **559 -> 18**, but true positives
fall **253 -> 193**, so recall drops **99.22% -> 75.69%**. Orange-circle recall drops
**92.40% -> 70.40%**; orange-square precision remains **66.67%** with **84.31% recall**.
The other three classes meet development thresholds. **Rejected, not selected or deployed.**
This shows that a crop verifier can reject many learned-detector false positives, but the
current verifier also rejects/relabels too many real targets; it cannot be assumed to transfer
from OpenCV proposals without further qualification. It is not a two-model Pi runtime claim.

## Checked checkpoint: epoch 7

`interim-g/development-640.json`, unchanged confidence 0.25, all 595 development frames:

| Class | Precision | Recall | Provisional class target |
| --- | ---: | ---: | --- |
| Orange circle | 94.86% | 66.40% | Fail: recall |
| Orange square | 83.04% | 69.61% | Fail: both |
| Orange triangle | 97.28% | 89.05% | Fail: recall |
| Yellow circle | 94.41% | 95.60% | Meets development target |
| Yellow square | 71.30% | 90.59% | Fail: precision |
| Yellow triangle | 98.33% | 99.16% | Meets development target |

Aggregate orange **91.89% precision / 74.35% recall**, yellow **82.81% / 94.00%**. Overall
**FAIL**. Lower validation loss and higher mAP50-95 did not imply passing at the fixed
operating point; checkpoints continue to trade precision against recall early in training.
The baseline run remains active and its planned maximum/early-stopping rule are unchanged.
Neither the verifier cascade nor reviewed dataset is silently substituted into this run.

A bounded CPU/MPS inference comparison on the six highest-detection-count epoch-7 validation
frames is retained in `interim-g/backend-comparison.json`. Both backends produce identical
counts/classes, matched box IoU >=0.999995, and maximum confidence difference <0.000002 with
the complete CPU NMS adapter. This limited check found no backend discrepancy explaining
the measured errors; it is not a general parity or Pi-runtime proof. No training settings changed.

## Checked checkpoint: epoch 8

`interim-h/development-640.json`, full 595-frame CPU check, confidence 0.25:

| Class | Precision | Recall |
| --- | ---: | ---: |
| Orange circle | 84.38% | 86.40% |
| Orange square | 81.17% | 95.10% |
| Orange triangle | 81.62% | 95.02% |
| Yellow circle | 78.46% | 96.23% |
| Yellow square | 84.05% | 84.71% |
| Yellow triangle | 84.89% | 99.16% |

**FAIL:** every class misses precision; orange circles and yellow squares also miss recall.
Orange aggregate **82.44% precision / 91.76% recall**, yellow **82.40% / 91.37%**.

`scripts/compare_goal_checkpoints.py` now compares entire raw-detector checkpoints at the
same standard, input size, backend and exact validation truth/frame set. It recomputes metrics,
rejects duplicate frames/checkpoints and verifier-cascade reports, and ranks by the **weakest
required class precision or recall divided by its required value**, using macro F1 only to
break ties. Insufficient class counts cannot pass. CLI reports must match the recorded standard
and its checksum. The comparison never promotes a model or opens the reserved test.

`comparison-through-epoch8-bound.json` ranks epoch 8 first: its weakest value is **78.46%**,
versus epoch 7's **66.40%**. None passes. This criterion follows the all-six-class goal and
does not cherry-pick classes from different epochs. The trainer's mAP checkpoint ranking is
unchanged; this separate, fixed-point comparison supports eventual whole-model selection.

Verification so far: **104 Python tests pass**, 2 detector-only tests skip in the Keras environment
and **pass separately in `.detector-venv`**, Ruff passes, four existing TFLite warnings.
Tests cover normalized/clipped tiny boxes, reserved-test exclusion, output class order,
confidence filtering, duplicate false positives, wrong-shape errors, sample sufficiency,
one-to-one matching, fixed size reporting, malformed/nonfinite predictions and complete NMS.
No viewer/browser changes in this experiment.
