# D-SHIFT

**Distilled Spatial-context Hybrid Inference with Freshness-aware Tracking**

D-SHIFT is a resource-constrained hybrid vision project for **Defend the Republic
(DTR)** balloon blimps. It combines OpenCV region detection, custom distilled
neural classifiers, and bounded temporal tracking on the **original Raspberry Pi
Zero W (ARMv6)**, not the Zero 2 W.

## What D-SHIFT means

- **D — Distilled:** compact crop classifiers learn from offboard MobileNetV4-Conv-Small teachers.
- **S — Spatial-context:** the context student preserves spatial layout and uses surrounding visual features.
- **H — Hybrid:** classical color-mask and region detection supplies candidates for neural classification.
- **I — Inference:** task-specific classifiers identify goal color/shape or balloon class and reject background.
- **F — Freshness-aware:** cached classifications have bounded lifetimes; stale labels are not retained indefinitely.
- **T — Tracking:** template-based position updates reduce repeated neural classification between scans.

## Architecture and model identity

The current goal research pipeline is:

```text
Pi camera → orientation/color conversion → OpenCV candidate regions
          → distilled crop CNN → duplicate suppression + bounded temporal tracking
          → image-space observations with class scores and freshness information
```

**D-SHIFT** names the complete perception project. **D-SHIFT Context-16K** names
the current 16,183-parameter goal-model accuracy reference: a custom 64×64 RGB
CNN with a context convolution, 2×2 spatial pooling, and seven output classes
(orange/yellow circle, square, triangle, plus background). It is distilled from
MobileNetV4, but its architecture is **not MobileNetV4**. The separate balloon
classifier and experimental 8,279-parameter separable goal student remain distinct
artifacts; the latest context-model benchmarks do not qualify the balloon branch.

Training uses native **Keras 3 / TensorFlow** on a Mac. Students are exported as
FP32 and INT8 TensorFlow Lite models. FP32 is faster for the measured models on
this Pi; an isolated ARMv6-targeted TensorFlow Lite build accelerates inference
without replacing the system library. These are measured research configurations,
not a claim that every viewer or Pi entry point uses the newest model/runtime.

D-SHIFT is a **perception research and integration pipeline**, not a completed
flight controller. Small/orange-target detection and held-out arena validation
remain open. Recognition and tracking do not establish metric distance, balloon
possession, a traversable goal opening, or permission to command motors.

The project was previously titled **DTR MobileNetV4-Conv-Small**. The existing
`dtr` Python package/CLI, model artifact names, and local `mobilenetv4-small`
folder remain unchanged for compatibility with saved environments and experiments.

Read the [complete project history](docs/PROJECT_HISTORY.md) for the development
timeline from choosing MobileNetV4 through data, training, Pi testing, rejected
experiments, and the October 6 ARMv6 runtime rebuild.

## Status and boundaries

- **Updated pixel comparison (2026-10-06):** [static, paced, and live-camera results](docs/COMPETITION_COMPARISON.md).
  The unchanged hollow-goal pixel method and context FP32 student were tested on
  the actual Pi. Paced generated replay delivered 9.85 versus 7.93 FPS; native-size
  desk-camera passes delivered 2.59 versus 3.31 FPS, respectively. Static yellow
  recall was 10.13% versus 90.81% at common 320×240 input. Different configurations
  and evidence scopes apply; these are not competition qualification results.

- **TensorFlow Lite rebuild (2026-10-06):** [ARMv6 build and actual Pi results](docs/TFLITE_REBUILD.md).
  The isolated 2.20.0 runtime passed all six reference-model checks. Context FP32
  crop inference averaged 23.64 ms versus 63.82 ms installed (2.70x throughput).
  The same context tracking replay measured 9.01–9.06 vs 5.40–5.43 processing FPS,
  excluding camera capture. The system library and original demo are untouched;
  select the candidate explicitly per process for further integration testing.

- **CV refinement (2026-10-06):** [real negatives and Pi optimization trials](docs/CV_REFINEMENT.md).
  Reviewed training backgrounds reduced warm-student orange-square false positives
  from 150 to 95, but overall development F1 regressed, so no replacement model
  was selected. An exact RGB lookup was slower on the Pi and remains disabled.
  Native tracking-difference calculations are now the local temporal default:
  paired Pi replay throughput improved 3.6–4.6%, with unchanged detection counts
  and exact fixed-clock output parity (not live-camera FPS). The
  existing viewer and original Pi demo are unchanged.

- **Planned goal recovery:** [opposing-goal search heuristic](docs/NAVIGATION_NOTES.md).
  If our goal is lost or not found during delivery, use confirmed opposing goals
  to guide a bounded search away from them. Recorded requirement, not implemented.

- **CV progress (2026-10-05):** [smaller context model and Pi measurements](docs/CV_PROGRESS.md).
  A trained-weight-initialized separable student uses 8,279 parameters and measures
  43.65 ms/crop on the actual Pi versus 64.50 ms for the 16,183-parameter context
  model. The four-crop synthetic-motion Pi replay measures 6.29 vs 5.17 processing
  FPS (not live camera FPS). Development full-frame F1 is 62.79% vs 63.23%, but square false positives
  regress, so it remains an opt-in speed experiment. The grayscale edge fallback
  regressed and stays disabled. The viewer now exposes tentative shape-only labels.

- **Shape/color uncertainty:** [separate goal evidence](docs/GOAL_UNCERTAINTY.md)
  retains strong shape scores when color is ambiguous, without changing accepted
  labels. An opt-in unknown-color investigation audit added 11 false positives
  and no true positives on 595 development frames; it is not a pursuit policy.

- **Open student research (2026-10-05):** five controlled/follow-up student runs
  produced a 16,183-parameter context/layout candidate with 98.31% crop accuracy
  and 63.23% six-class full-frame F1 (previous candidate: 51.57%) on reused
  development data. [Research results and next experiments](docs/STUDENT_RESEARCH.md)
  include oracle diagnostics, source-detail ablations and a prepared Pi benchmark.
  Pi crop latency is now measured in the newer CV-progress report; no existing
  integration model or viewer model was replaced.

- **Pi optimization (2026-10-05):** the same tiny goal student now has a 25-epoch
  checkpoint, native FP32 support, and bounded temporal classification/tracking.
  [Measured experiments and run commands](docs/PI_OPTIMIZATION.md) separate
  595-frame accuracy, synthetic-motion speed/recall, and actual camera throughput.
  This is an isolated integration candidate; the original Pi implementation,
  frozen comparison, and existing web viewer are preserved. Orange goals remain weak.

- **Pi comparison work (2026-10-05):** real-data students are trained and exported: balloon
  10,640 bytes, goal 10,880 bytes, both full INT8. [Comparison protocol and results](docs/PI_COMPARISON.md)
  distinguish crop accuracy, full-frame detection, and actual Pi timing. These are research
  candidates, not flight-qualified models. Existing web viewer remains on its teachers.

- **Close-range follow-up paused at a saved epoch:** a [four-epoch exposure experiment](docs/DETECTOR_CLOSE_EXPOSURE.md)
  repeats 349 train-only close-range frames across all six classes. Validation is unchanged;
  3,997 entries still represent only 2,950 unique training frames. Epoch 1 improves yellow-square
  recall to 91.37%, but orange-triangle precision (77.11%) and yellow-circle precision (85.00%)
  regress. It is not selected; epoch 2 was saved. Priority has shifted to Pi integration and comparison.

- **Goal-localization refinement completed, standard unmet:** [isolated YOLO11n experiment](docs/DETECTOR_PROGRESS.md)
  learns boxes directly for both colors. Provisional per-class precision/recall targets are
  recorded; no passing detector result or Pi qualification yet. After the user freed storage
  (24 GiB available), training resumed from epoch 8 with a 10-GiB free-space guard.
  Baseline epoch 17 was its strongest whole-checkpoint candidate: both color aggregates exceed 90%,
  but orange-triangle precision (86.55%) and yellow-square recall (88.63%) still fail.
  The baseline stopped normally at epoch 21; final/native-best checkpoints also fail.
  A separate [reviewed-label refinement](docs/DETECTOR_REFINEMENT.md) completed 12 epochs from epoch 17,
  with lower learning rate and no mosaic. Refinement epoch 2 now ranks strongest, but
  orange-triangle precision (86.88%) and yellow-square recall (89.41%) still fail.
  Epochs 3–12 and finalized native best rank lower. A 960-input experiment also failed.
  No training process remains active for that completed run. The [error/coverage review](docs/GOAL_ERROR_FOLLOWUP.md)
  identifies concentrated close-range misses and box/annotation issues for the next experiment.
  The full standard remains unmet.
  No checkpoint is promoted. Training-label corrections leave validation unchanged.
  Existing viewer is unchanged.
  Native memory growth required a controlled restart from epoch 13. Training now uses
  one-epoch process boundaries (exit 75 verified, then resume) with saved early-stopping state.
  Random-stream persistence is unit-tested and verified live across the epoch-15/16 boundary.
  Full-precision raw-weight/optimizer continuation is unit-tested and verified live across
  epoch 18/19, with exact saved/restored state comparison before the first training batch.

- **Orange remains required:** the [small-region experiment](docs/ORANGE_REGION_PROGRESS.md)
  improved orange circles but regressed triangles and was rejected. New train-only regression
  checks protect every goal class and size stratum. Viewer defaults remain unchanged.

- **Goal follow-up:** [stage-level audit](docs/GOAL_STAGE_PROGRESS.md) now separates box,
  budget and classifier failures. Small-target localization dominates orange misses. Two
  proposal/resolution experiments were rejected; selected viewer accuracy and defaults
  remain unchanged. 57 tests pass; reserved test remains untouched.

- **Earlier teacher result:** the [balloon component filter](docs/COMPONENT_PROGRESS.md) improves development
  precision/recall to **77.8%/81.8%** without losing true detections in the same validation set.
  Goal remains **72.3%/54.6%**. Same teachers and 12-candidate limits; test untouched. A larger
  goal budget is optional desktop diagnosis, not the default or a Pi qualification.

- **Start visual testing:** see [camera/image testing](docs/TESTING.md). The viewer now defaults
  to the proposal-adapted real V10 teachers and accepts saved images. The improved box finder still produces
  frequent false detections; this is a debugging build, not an autonomous flight system.
- This is an executable research/training pipeline, not a flight controller.
- Public ImageNet V4 weights are imported into Keras with numerical parity checks.
- Balloon and goal data preparation, fine-tuning, distillation, INT8 conversion,
  evaluation, crop inference, and bounded OpenCV video replay are implemented.
- Synthetic smoke data verifies software plumbing, NOT balloon/goal accuracy.
- 2026-10-03: authorized DTR COCO V10 and V11 downloads completed. V10 retains more
  detail (640x640 vs 180x180) and is selected, without mixing the overlapping versions.
  5,492 frames survive duplicate/conflict/group checks; both task datasets are ready.
  Both real-data V4 teachers completed: 99.915% balloon and 96.447% goal **validation crop**
  accuracy. The existing full-frame box proposer still misses many targets; no distillation yet.
  See [real-data status and teacher results](docs/REAL_DATA_STATUS.md). Recording-session
  independence and competition performance are not established by this public dataset.
- 2026-10-05: train-only proposal adaptation completed. On the same 595 development-validation
  frames, balloon precision/recall is 42.6%/81.8%; goal is 48.0%/54.8%. Better than the original
  teachers under the same proposal settings, but still frequent false detections. Test untouched.
  New inference pairs: `runs/{balloon,goal}-proposals-20261005/teacher.{keras,json}`.
- Follow-up: same-class duplicate suppression improves current balloon precision/recall to
  **44.4%/81.8%**, goal to **72.3%/54.6%**, on those same development-validation frames.
  Viewer shows suppressed candidates and shorter rejected-box labels. Orange-mask experiments
  failed train-only checks and were not selected. No teacher retraining or distillation in this step.
- Native ARMv6 runtime and bounded camera benchmarks have now run on the actual Pi;
  see [Pi handoff](docs/PI.md). These checks do not establish flight readiness.
- No SSH, motors, UART commands, public uploads, or cloud inference occur implicitly.
- Exported models are unapproved by default. Inference requires an explicit
  `--allow-unvalidated` flag for research use. A confidence score is not a calibrated probability.

## Install (Apple Silicon, Python 3.11)

Clone the private repository with an authorized GitHub account:

```bash
git clone https://github.com/CaswellRyder/d-shift.git
cd d-shift
```

Create the isolated `.venv` in the checkout. Nothing is installed into system Python.
Datasets, trained weights, experiment outputs, and compiled runtime bundles are
not included in Git. See the data preparation, training, and runtime build guides
below to reproduce them; existing local artifacts remain untouched.

```bash
uv sync --extra pretrained --extra dev --extra metal
source .venv/bin/activate
dtr doctor
```

`uv.lock` pins the complete environment. On non-Mac hosts omit `--extra metal`.
Use `dtr --cpu ...` for diagnostic runs; Metal is a Mac GPU plugin, not CUDA or the
Apple Neural Engine. The teacher does not run on the Pi.

## First: inspect a completed run

See `runs/balloon-smoke-002/report.json` and `runs/goal-smoke-001/report.json` if present.
Only a completed report proves a run finished; `status.json` records running/failed states.

```bash
dtr --cpu predict \
  --model runs/balloon-smoke-002/student.int8.tflite \
  --image data/smoke-balloon/images/test/green_balloon/0000.png \
  --allow-unvalidated
```

These predictions are from a synthetic smoke exercise and are not useful flight evidence.

## Reproduce the end-to-end smoke run

For current teacher development, use **`train-teacher`**, described below and in
[the training guide](docs/TRAINING.md). It stops before distillation and leaves the test
split untouched. The older **`train`** command below runs the entire pipeline, including
distillation, quantization, and test evaluation; it is not the teacher-only workflow.

Always use new output directories; commands refuse to overwrite existing runs/datasets.
`make smoke` runs both tasks with timestamped directories. The individual commands are:

```bash
# One-time ImageNet checkpoint download + PyTorch-to-Keras import + parity assertion:
dtr --cpu pretrained

dtr synthetic --config configs/balloon.json --output data/balloon-demo
dtr train --config configs/balloon.json \
  --manifest data/balloon-demo/manifest.json \
  --pretrained artifacts/pretrained/mobilenetv4_conv_small.keras \
  --output runs/balloon-demo --smoke

dtr synthetic --config configs/goal.json --output data/goal-demo
dtr train --config configs/goal.json \
  --manifest data/goal-demo/manifest.json \
  --pretrained artifacts/pretrained/mobilenetv4_conv_small.keras \
  --output runs/goal-demo --smoke
```

If the pretrained artifact already exists, skip `dtr pretrained`. Its companion JSON
contains provenance, SHA-256 and output-parity evidence. This one-time importer uses
`timm`/PyTorch; all subsequent learning and distillation uses native Keras.

## Get real DTR data

See [public sources and access limitations](docs/DATA_SOURCES.md).
The current local datasets are `data/roboflow-dtr-v10-grouped/{balloon,goal}/manifest.json`.
See [training](docs/TRAINING.md) for their run commands. To reproduce from fresh paths,
`scripts/download_dtr.py` prompts for a key without saving it; `scripts/prepare_roboflow_dtr.py`
audits the exports and writes review sheets. Inspect them before running its `--prepare`
and `--review-backgrounds` modes. These scripts refuse to overwrite existing data.

For another source, the general workflow is:
Download an authorized **COCO JSON** export and extract it to `data/raw/dtr`:

```text
data/raw/dtr/
  train/_annotations.coco.json  + image files
  valid/_annotations.coco.json  + image files
  test/_annotations.coco.json   + image files
```

Prefer splitting by recording/session, not adjacent video frames. Supply a JSON mapping:

```json
{
  "train/frame001.jpg": {"session": "arena-day1-recording1", "split": "train"},
  "valid/frame002.jpg": {"session": "arena-day2-recording1", "split": "val"},
  "test/frame003.jpg": {"session": "arena-day3-recording1", "split": "test"}
}
```

Each source image needs an entry. All frames from one session must have the same split.
If original sessions are unavailable, `--trust-source-splits` explicitly permits the public
export's splits but marks session independence **UNVERIFIED**. Exact duplicate source pixels
across splits are rejected regardless. This does not detect all near-duplicate video frames.

```bash
dtr prepare-coco --config configs/balloon.json --source data/raw/dtr \
  --sessions data/sessions.json --output data/balloon
dtr prepare-coco --config configs/goal.json --source data/raw/dtr \
  --sessions data/sessions.json --output data/goal

dtr validate --config configs/balloon.json --manifest data/balloon/manifest.json
dtr train-teacher --config configs/balloon.json --manifest data/balloon/manifest.json \
  --pretrained artifacts/pretrained/mobilenetv4_conv_small.keras --output runs/balloon-real-001
dtr train-teacher --config configs/goal.json --manifest data/goal/manifest.json \
  --pretrained artifacts/pretrained/mobilenetv4_conv_small.keras --output runs/goal-real-001
```

Unknown labels are reported, NOT guessed. Generic `Ball`/`Balloons` labels cannot identify
color and are omitted from the colored-balloon classifier. Background sampling excludes
all annotated objects, including unknown categories; incomplete annotations can still
produce false background labels, so inspect crops. Add difficult OpenCV proposals and
Pi-camera footage before claiming real-world performance. Check current competition colors.

## Exact learning pipeline

1. V4-Conv-Small width 1.0, 96×96 RGB, ImageNet normalization embedded in the model.
   Topology follows V4 Conv-Small; symmetric padding and BN epsilon reproduce the selected
   timm checkpoint, rather than silently mixing TensorFlow `same` padding with timm weights.
2. Balloon classes: background / green / purple. Goal classes: background + six color/shape
   combinations. Class order lives in configs and every exported metadata file.
3. Train new head (Adam 1e-3, 5 epochs); fine-tune backbone (Adam 1e-5, 15 epochs).
   BatchNorm statistics remain frozen; early stopping restores best validation loss.
4. Distill into a 64×64 RGB student: Conv3×3 stride2 [8,16,32] + ReLU, global average pooling,
   logits. `0.5 CE + 0.5 × 4² × KL(softmax(teacher/4) || softmax(student/4))`.
   Teacher frozen, student Adam 1e-3, up to 25 epochs. Resize both from the same original crop;
   brightness/flip transforms are paired. No hue augmentation.
5. Export full INT8 using up to 500 shuffled TRAINING crops. Integer input/output, no float
   fallback. Wider integer accumulators are normal. The exporter freezes TF variables first
   to avoid TF 2.18/Keras 3 `READ_VARIABLE` calibration failures.
6. Report teacher/FP32/INT8 test metrics and Mac crop latency. No automatic deployment approval.

The 96×96 teacher resolution and training hyperparameters are starting choices, not validated
optima. Imported ImageNet behavior is parity-checked at that resolution, not requalified on ImageNet.

## Artifacts in every successful run

Teacher-only runs contain `teacher.keras`, `teacher.json`, optimizer-bearing epoch
checkpoints, resumable `state.json`, validation predictions/metrics, logs, config and
provenance. They do **not** produce students or INT8 exports. See [training](docs/TRAINING.md).

The legacy end-to-end `train` command produces:

- `teacher.keras`, `student.keras`
- `student.int8.tflite` and `student.int8.json` (class order, preprocessing, quantization, SHA)
- `config.json`, `provenance.json`, `status.json`, `report.json`
- CSV epoch logs and TensorBoard events for each training phase

## Observe a video (no actuation)

### Live webcam in your browser

On your **MacBook**, from the project directory and activated Python 3.11/3.12 environment:

```bash
conda activate ENGR-E399-CV
python -m pip install -e ".[dev]"
python webcam_app.py --allow-unvalidated
```

Open `http://127.0.0.1:8765` if it does not open automatically. Click **Start camera**
and allow browser camera access. **Demo** exercises actual inference on clearly labeled
moving synthetic shapes without opening the camera. Select Balloon or Goal while stopped.
The proposal-adapted teachers are loaded by default; use `--balloon-model` and `--goal-model`
to point to other matching model/metadata pairs. `--no-browser` disables automatic opening.

To inspect the newly trained **teachers**, pass their `.keras` files instead:

```bash
python webcam_app.py --allow-unvalidated \
  --balloon-model runs/balloon-teacher-smoke-20260929/teacher.keras \
  --goal-model runs/goal-teacher-smoke-20260929/teacher.keras
```

These particular runs use synthetic fixtures, not real DTR data. Copy both `teacher.keras`
and its `teacher.json` to the MacBook along with the updated code. Teacher inference needs
the training TensorFlow/Keras environment; this is not the Pi runtime.

- Dashed amber outlines: OpenCV color candidates; solid green/gray inset: accepted/rejected.
- Labels and scores: actual classifier output (V4 teacher or INT8 student, shown in the header),
  not ground truth or calibrated confidence.
- Blue trails and IDs: simple greedy IoU association across analyzed frames, including rejected
  candidates. Not Kalman prediction, robust re-identification, capture confirmation or navigation.
- Color mask: white pixels passed the same HSV/morphology filter used for proposals.
- Processing time excludes capture; round trip includes encoding, local transport and inference.
- Camera target and source-image ceiling: **2592 × 1944**. Larger sources are downscaled to
  fit; smaller sources are not upscaled. Actual source and capped dimensions are displayed.
  Camera mode requests are preferences, not proof that the webcam supplied 5 MP.
- Sources are then letterboxed to the existing **320 × 240 processing frame**, followed by
  model-sized crops. This does not run proposals at 5 MP. Non-4:3 sources are no longer stretched;
  their results can differ from earlier viewer tests.
- Set **Processing FPS cap** before starting (default **5**, selectable **0.2–30**).
  Camera-reported FPS and measured delivered FPS are shown separately. Slow inference lowers
  the delivered rate; there is one request in flight and no queued-frame backlog or catch-up burst.
  Desktop pacing is not a Pi benchmark; this viewer still uses its selected models, not an
  automatic switch to the Pi temporal pipeline. See [testing details](docs/TESTING.md#resolution-and-fps-controls).
- Boxes overlay the exact
  JPEG frame analyzed, not a newer video frame. Tracking resets when restarted and expires after
  0.75 seconds without a match. Fast movement, occlusion and overlapping objects can switch IDs.

Stop releases the camera and clears stale overlays. Hiding/closing the tab stops capture;
Ctrl+C stops the Python server. No recording, microphone access, cloud uploads or flight commands.
This is a **loopback-only research utility**, not a publicly deployable web service.
All frames stay in browser/Python memory on the same computer. Run it locally, not via SSH
on another machine. Camera permission is browser-controlled; on macOS check Privacy & Security
→ Camera if access is denied. The demo does not prove webcam hardware works.

No Node.js, Flask, GUI-enabled OpenCV or additional Python packages are required.
If the model files are missing, copy `runs/balloon-smoke-002/student.int8.*` and
`runs/goal-smoke-001/student.int8.*` from the training computer too.

### Recorded-video replay

```bash
dtr --cpu replay --model runs/balloon-real-001/student.int8.tflite \
  --video data/arena.mp4 --output runs/arena-observations.jsonl --max-frames 100 \
  --allow-unvalidated
```

Replay resizes frames to 320×240, proposes up to 3 regions using provisional HSV ranges,
and runs only the requested specialist. No tracker, calibrated bearing/range, goal-opening
clearance check, mission controller, or motor output is claimed. Replay may miss objects
that OpenCV never proposes. It is a perception integration harness, not autonomous flight.

## Original Pi Zero gate

Copy `runtime.py`, the student `.tflite`, its `.json`, and a test image ONLY after obtaining
a compatible ARMv6 interpreter plus NumPy/Pillow. Do NOT install this training environment
on the Pi or assume modern `tflite-runtime` wheels support it. See [Pi handoff](docs/PI.md).
The student's memory/latency must be measured with the camera and logging active.

## Development

```bash
uv run --extra dev pytest -q
uv run --extra dev ruff check src tests
.venv/bin/tensorboard --logdir runs --host 127.0.0.1 --port 6006
```

See [verification receipt](docs/VERIFICATION.md) for checks actually performed on this machine.
See [third-party notices](THIRD_PARTY_NOTICES.md) for model/data attribution.
