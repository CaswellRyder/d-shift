# Local verification receipt — 2026-09-28

**Current data update (2026-10-03):** authorized real DTR V10/V11 exports have now been
downloaded, audited and prepared for both specialists. Both real-data teachers completed;
Ruff and 28 tests passed. See [current data/training receipt](REAL_DATA_STATUS.md), including
the measured proposal-coverage limitation and reserved test split.
The dated smoke-test evidence below is retained as history, not the current real-data status.

## Verdict

The local engineering pipeline executes successfully. No real DTR dataset has been
trained, and no model is approved for flight or qualified on an original Pi Zero.

## Environment

Apple Silicon M4 Max, arm64 macOS, isolated Python 3.11.15 environment.
TensorFlow 2.18.1, Keras 3.8.0, TensorFlow Metal 1.2.0. Training runs detected and
used the Metal GPU. TFLite inference measurements are Mac CPU crop measurements.
Dependencies are pinned in `uv.lock`; setuptools 75.8.2 preserves the `pkg_resources`
dependency needed by the pinned TensorBoard version.

## Evidence

- Public ImageNet checkpoint imported from timm into native Keras: 93 mapped layers.
  Three deterministic RGB inputs passed numerical parity at 96×96;
  maximum absolute logit error was 0.0000686646, tolerance 0.0005.
  Receipt: `artifacts/pretrained/mobilenetv4_conv_small.json`.
- `runs/balloon-smoke-002/report.json`: completed V4 head training, fine-tuning,
  distillation, integer conversion, test evaluation, and local inference.
- `runs/goal-smoke-001/report.json`: completed the same stages for seven goal classes.
- `make smoke` also completed with exit 0 after final integration changes, producing
  `runs/balloon-smoke-20260928T223432Z` and `runs/goal-smoke-20260928T223432Z`.
- Both exports have integer input/output and no floating-point tensors; runtime
  inference and artifact checksum checks succeed.
- TensorBoard responds on `http://127.0.0.1:6006` and exposes training/validation
  scalar tags for all three stages of both tasks. Restart with `make viewer`.

| Synthetic smoke result | Balloon | Goal |
|---|---:|---:|
| INT8 artifact bytes | 10,640 | 10,880 |
| Calibration training crops | 36 | 84 |
| Held-out synthetic crops | 36 | 84 |
| Teacher accuracy | 75.0% | 94.05% |
| FP32 student accuracy | 66.67% | 14.29% |
| INT8 student accuracy | 66.67% | 14.29% |

Each training stage ran only one epoch. The goal student is at chance performance;
these runs prove execution, not successful task learning. INT8 agreement with a weak
student does not establish usefulness. Artifact size does not represent working RAM.

## Regression checks

`make test` passed lint and all 12 tests. Coverage includes model serialization, split leakage,
manifest tampering, matching crop preprocessing, frozen-teacher distillation,
integer export, inference approval/checksum checks, bounded OpenCV proposals,
COCO import policy and ingestion, and offline replay. Replay is observational and
cannot issue flight commands. Benchmark and replay require explicit research opt-in.

## Known limitations and retained failure

- `runs/balloon-smoke-001/status.json` retains the first converter failure.
  Freezing model variables before conversion fixed the resource-variable calibration
  error. Later successful runs use this fix; the failed run was not relabeled.
- TensorFlow emits converter/interpreter deprecation warnings. The pinned converter
  uses an internal variable-freezing API; upgrades require running these checks again.
- Public DTR projects were located, but the tested Roboflow COCO download endpoint
  returned HTTP 401 without an authorized API key. See `DATA_SOURCES.md` for manual
  export routes and attribution. No credentials or access controls were bypassed.
- Original ARMv6 interpreter availability, live Pi camera capture, proposal recall,
  camera-loop p95 latency, RAM, tracking, goal geometry, MCU communication and flight
  behavior remain unverified or outside this model-training project's scope.

Next: acquire an authorized public COCO export, inspect labels/session splits,
train both specialists, and evaluate held-out real recordings before Pi trials.

## Teacher-only development verification — 2026-09-29

- Added `train-teacher`, `evaluate-teacher`, and `predict-teacher`. Teacher training stops
  before distillation; model selection uses validation only. Full epoch checkpoints retain
  optimizer state, with checksum/config/dataset/version guards and exclusive run locking.
- `make test`: lint and 21 tests passed, with four existing TFLite conversion/deprecation
  warnings. Lifecycle tests verify optimizer iteration restoration, interruption before the
  first epoch, immutable resume inputs, research opt-in, and explicit test evaluation.
- Actual V4 balloon smoke: one head epoch, intentional epoch-budget pause, resume, one
  fine-tuning epoch. Goal smoke: one head and one fine-tuning epoch. Both completed with
  `synthetic_training: true`, `deployment_approved: false`, and `distillation_started: false`.
  Run directories: `balloon-teacher-smoke-20260929` and `goal-teacher-smoke-20260929` under `runs/`.
- Balloon synthetic validation: 36 crops, accuracy 0.7222, macro-F1 0.6639. Goal synthetic
  validation: 84 crops, accuracy 1.0. These tiny generated fixtures do not establish DTR
  performance. The separate balloon `test-explicit-smoke.json` verifies explicit evaluation
  on 36 synthetic test crops, not real-world generalization.
- Local HTTP viewer check used real `.keras` teachers and generated 320x240 JPEGs. Both task
  configs identified `keras_teacher`; frame responses included predictions, candidate boxes,
  track IDs, and mask data. No camera was activated. JavaScript syntax check passed.
- Rendered browser inspection was unavailable (no browser exposed to automation); webcam
  hardware and the new header's rendered appearance were not verified in this turn.
- No real dataset was available locally; the checked public COCO export returned HTTP 401.
  No real-data training, new distillation run, Pi deployment, or laptop synchronization occurred.

See [teacher training instructions](TRAINING.md) for real-data preparation, resume, evaluation,
and teacher-mode visual inspection.
