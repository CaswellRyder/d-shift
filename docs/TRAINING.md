# Teacher-first training

Use `dtr train-teacher` for the current development stage. Unlike the legacy `dtr train`,
it never starts distillation, exports INT8, or automatically evaluates the test split.
All commands below run from the project root in the installed Python 3.11 environment.

## Data gate

Update 2026-10-03: authorized Roboflow V10/V11 exports acquired. V10 was audited and
prepared into `data/roboflow-dtr-v10-grouped/{balloon,goal}/manifest.json`. Both tasks have
all configured classes in every split. See [real-data status](REAL_DATA_STATUS.md) and
[sources](DATA_SOURCES.md). Upstream random frame splits were replaced by inferred filename
groups/time blocks, a temporal guard, and cross-split duplicate filtering. These are NOT
verified independent recording sessions. Representative target and background crops were
visually reviewed, not every annotation. Add difficult actual camera proposals and separately
recorded Pi footage before qualification.

Every configured class must occur in train, val, and test. Keep recording sessions disjoint.
Do not tune with the test set. Synthetic data requires `--smoke` and is only a software test.
Crop accuracy alone cannot prove full-frame proposal recall, tracking quality, or flight readiness.

## Current real-data teacher commands

```bash
dtr validate --config configs/balloon.json --manifest data/roboflow-dtr-v10-grouped/balloon/manifest.json
dtr train-teacher --config configs/balloon.json \
  --manifest data/roboflow-dtr-v10-grouped/balloon/manifest.json \
  --pretrained artifacts/pretrained/mobilenetv4_conv_small.keras \
  --output runs/balloon-dtr-v10-20261003

dtr validate --config configs/goal.json --manifest data/roboflow-dtr-v10-grouped/goal/manifest.json
dtr train-teacher --config configs/goal.json \
  --manifest data/roboflow-dtr-v10-grouped/goal/manifest.json \
  --pretrained artifacts/pretrained/mobilenetv4_conv_small.keras \
  --output runs/goal-dtr-v10-20261003
```

The verified ImageNet V4 backbone receives a new task head. Defaults: RGB 96x96,
head Adam 1e-3 for up to 5 epochs, backbone Adam 1e-5 for up to 15 epochs, batch 16,
BatchNorm frozen, early-stopping patience 5. Config may override `head_lr`, `finetune_lr`,
`patience`, and `min_delta`. These are starting settings, not validated optimal choices.
The final model is selected by validation loss across both phases, including the best
head-only model if fine-tuning is worse. Threshold 0.8 is provisional, not calibrated.

Use a new output directory for each experiment. The pretrained `.keras` must have its
matching parity/checksum `.json`. Training uses the local Mac, not the Pi or a cloud GPU.

## Proposal adaptation (2026-10-05)

`--initial-teacher` warm-starts the **complete existing teacher**, including its learned
classifier. It is mutually exclusive with `--pretrained`, which imports only a backbone
and creates a new classifier. The initial teacher's checksum, task, class order and input
size must match. Both paths retain epoch-boundary resume.

The reviewed adaptation manifests are
`data/dtr-proposals-reviewed-20261005/{balloon,goal}/manifest.json`.
Do not train the raw `data/dtr-proposals-20261005/` candidates: missing upstream annotations
make some supposed background crops actual targets. Those manifests are quarantined.
AI visual review admitted only 161 balloon-task and 176 goal-task hard negatives from
256 inspected candidates per task. Each admitted negative receives four training entries
(oversampling, not four independent images). New positives use upstream IoU>=0.5 labels,
with representative visual inspection, not a complete human audit. One conflicting goal
training source was excluded. Original validation and test crop records remain unchanged.

```bash
dtr train-teacher --config configs/balloon-proposals.json \
  --manifest data/dtr-proposals-reviewed-20261005/balloon/manifest.json \
  --initial-teacher runs/balloon-dtr-v10-20261003/teacher.keras \
  --output runs/balloon-proposals-20261005
dtr train-teacher --config configs/goal-proposals.json \
  --manifest data/dtr-proposals-reviewed-20261005/goal/manifest.json \
  --initial-teacher runs/goal-dtr-v10-20261003/teacher.keras \
  --output runs/goal-proposals-20261005
```

These fixed adaptation settings use two head-only epochs at 1e-4, up to four backbone
epochs at 5e-6, batch 32, frozen BatchNorm and patience three. Checkpoints are still selected
by the original crop-validation loss. Promotion additionally requires comparing full-frame
validation with unchanged proposals and threshold; crop accuracy is not the deciding evidence.
Fresh output directories are required to repeat a run. No test inference or distillation.

## Pause and resume

Add `--epoch-budget 1` to stop after one completed epoch for an initial controlled check.
Resume with the same config, manifest, output directory, and smoke flag (if originally used):

```bash
dtr train-teacher --config configs/balloon.json \
  --manifest data/roboflow-dtr-v10-grouped/balloon/manifest.json \
  --output runs/balloon-dtr-v10-20261003 --resume
```

Each completed epoch saves the model and optimizer before atomically updating its state
pointer. Resume restores the last committed epoch, not an interrupted partial epoch.
Optimizer state is restored within a phase; switching phases intentionally creates a new
optimizer. RNG/augmentation stream position is not restored, so resume is not bitwise-identical
to uninterrupted training. Config, manifest checksum, TensorFlow/Keras versions, and checkpoint
checksum must match. Before the first checkpoint, the original pretrained path is still needed.
Completed runs cannot be resumed. Concurrent writers to the same run are refused.

Checkpoints are retained every epoch, so allow disk space for multiple full models plus
optimizer states. `state.json` is authoritative for checkpoint selection; an interrupted
CSV/TensorBoard update can lag it. Copy the complete run directory to move a resumable run.

## Inspect and evaluate

`report.json` records completion and validation metrics. `validation.json` records per-crop
predictions, confusion matrix, per-class precision/recall/F1, accepted errors, and a validation
threshold sweep. Do not interpret zero accepted predictions as perfect accepted precision.
`teacher.keras` plus `teacher.json` are the inference pair.

```bash
dtr evaluate-teacher --run runs/balloon-dtr-v10-20261003 --split val \
  --output runs/balloon-dtr-v10-20261003/validation-review.json
```

After choosing the model/settings using validation only, run a separate held-out test:

```bash
dtr evaluate-teacher --run runs/balloon-dtr-v10-20261003 --split test \
  --output runs/balloon-dtr-v10-20261003/test-final.json
```

Evaluation refuses to overwrite its output. On another machine, pass `--manifest` with the
relocated manifest (identical file checksum). The training report's `test_evaluated: false`
describes training itself; explicit evaluation creates a separate report rather than changing it.
Neither evaluation nor successful training authorizes deployment or automatically starts distillation.

## Visual inspection before distillation

See [current visual testing](TESTING.md) for saved-image testing and full-frame comparisons.
Defaults now point to the proposal-adapted teachers; explicit original paths remain supported.

```bash
python webcam_app.py --allow-unvalidated \
  --balloon-model runs/balloon-proposals-20261005/teacher.keras \
  --goal-model runs/goal-proposals-20261005/teacher.keras
```

Open the local viewer, choose a task, and use Demo or explicitly start the camera.
The header identifies the V4 teacher. Boxes originate from OpenCV candidates; the teacher
classifies crops, not the full image. IDs are simple display tracking, not flight navigation.
This viewer does not save footage or labels. The teacher runs on the Mac, not the Pi Zero.
The current proposal-coverage audit in [real-data status](REAL_DATA_STATUS.md) shows substantial
misses, especially orange goals. The viewer is for debugging, not proof that training crop
accuracy transfers to full-frame detection. Do not distill a classifier as a substitute for
fixing candidate coverage and collecting actual proposal hard negatives.

## Verified synthetic exercise (2026-09-29)

`runs/balloon-teacher-smoke-20260929` completed one head epoch, paused, resumed, then completed
one fine-tuning epoch. `runs/goal-teacher-smoke-20260929` completed both phases. Both used the
actual pretrained V4 implementation and synthetic fixture manifests; neither began distillation.
These artifacts verify execution, not useful DTR accuracy. Use their paths in the viewer only
for software inspection. They predate the real-data runs described above.
