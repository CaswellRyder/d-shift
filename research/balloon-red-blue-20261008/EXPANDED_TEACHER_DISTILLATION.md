# Expanded MobileNetV4 teacher and controlled distillation

## Verdict

The updated offboard MobileNetV4 Conv Small teacher improves full-scene red/blue
detection. Transferring that improvement to the 7,763-parameter Pi student is
incomplete and seed-sensitive. No candidate is deployment-approved, no Pi
runtime changed, no new Pi FPS was measured, and the 94% precision/recall goal
remains unmet. The original reserved test remains unevaluated.

## Training and separation

The teacher was warm-started from
`runs/balloon-red-blue-teacher-20261008/teacher.keras`, SHA-256
`9eaa9452f4ab33cf5631e1dec4e0f391025189f4a7236222ca3a60cea2e27236`.
The expanded manifest is
`data/balloon-red-blue-indoor-proposal-admit-20261008/manifest.json`, SHA-256
`2aeadeb64c5b484e9e6f242af53e4911f615768adc84309b251cc2d691e218fb`.
Its 510 training entries contain 198 background, 174 red and 138 blue views;
repeated crops are not independent images. Zero training source basenames begin
with `IMG_`. The entire EngDes2 IMG family stays out of training, calibration and
negative mining, and the original reserved test remains separate.

Teacher training now scopes pixel validation to train/val, retaining all-split
metadata leakage checks. Validation-only evaluation opens only val pixels;
explicit test evaluation remains a separate action. A regression test fails if
training or validation implicitly hashes a reserved test image.

The 96x96 teacher used the existing config: five head epochs followed by up to
15 fine-tuning epochs with early stopping. Thirteen epochs completed (five head,
eight fine-tune). Fine-tune epoch three had the lowest original eight-crop
validation loss (0.0201393) and was selected. The exported teacher SHA-256 is
`db57446ecfba29c29353924792908d9a762f5452f50664f66c43cfdcc0ff45fe`.
It accepts all five target validation crops correctly, versus three for the
original teacher. Eight crops are insufficient evidence for qualification.

The existing uncommitted proposal-builder and student augmentation changes were
preserved and used, not claimed or committed with this experiment. Student
training script SHA-256:
`51ab5e57c819a9f84a4d7b9dc5fe5cfab194d013b945a2be046ece7f145af10a`.
Each ignored run retains config, checkpoint identities, history and provenance.

## Full-scene teacher comparison

The twelve reviewed indoor development scenes contain 11 red and 12 blue
targets, including three negative frames. They are repeatedly inspected,
seen-environment development evidence, not independent capture sessions or
flight qualification. Operating point: confidence 0.8, same-class one-to-one
matching at IoU 0.5. With MSER and confirmed-parent suppression, TP/FP/FN are:

- Original teacher: red 10/3/1; blue 6/1/6.
- Expanded teacher: red 11/1/0; blue 10/1/2.

Expanded teacher precision/recall: red 91.7%/100%, blue 90.9%/83.3%.
Both remaining blue misses lack a sufficiently localized search proposal;
changing classifier confidence alone cannot recover them.

On the original four development photographs, the same search changes red
2/2/0 to 2/1/0 and blue 2/1/1 to 3/0/0. One red false positive persists.
On the separate eleven known clutter crops the expanded teacher accepts one,
versus zero for the original teacher: a regression, not an improvement.

## Matched student experiment

Two seeds (42 and 43), each with two freshly initialized separable-context
students: 64x64 RGB, 7,763 parameters, 25 epochs, Adam 0.001, no extra geometric
augmentation. The original eight-crop validation hard loss selects checkpoints;
indoor scenes do not select epochs. Conditions differ only in supervision:

- Distilled: 0.5 hard-label cross entropy plus 0.5 teacher KL, temperature 4,
  with temperature-squared scaling. Verified train-only teacher logits come
  from the expanded teacher. Seed 43 reuses the hash-bound seed-42 logit cache.
- Control: hard-label cross entropy only (alpha 1); no teacher supervision.

With MSER confirmed-parent suppression on the twelve indoor scenes, FP32:

- Seed 42 distilled: red 11/1/0; blue 8/2/4.
- Seed 42 control: red 11/1/0; blue 6/2/6.
- Seed 43 distilled: red 10/2/1; blue 8/3/4.
- Seed 43 control: red 10/0/1; blue 8/3/4.

Thus seed 42 improves blue recall from 50% to 66.7%, but seed 43 has no blue
improvement and adds two red false positives. Two seeds do not establish a
variance estimate or robust superiority. Compared with the prior augmented
seed-42 student, the new seed-42 distilled model removes one red false positive
but has unchanged blue counts; those runs also differ in augmentation, so that
comparison does not isolate distillation.

Seed-42 distilled INT8 matches its FP32 counts. Seed-42 control INT8 recovers one
additional blue target versus its FP32 export. Seed-43 distilled INT8 recovers
one additional red target versus its FP32 export; the control is unchanged.
Export-specific checks remain necessary.

On the original four scenes with the same suppression, both distilled seeds
give red 2/2/0 and blue 3/0/0. Seed-42 control matches; seed-43 control adds one
blue false positive. On eleven known clutter crops, distilled/control accept
11/10 for seed 42 and 8/7 for seed 43. These crops were not included in the
current training manifest, despite the evaluator's legacy `training_fit` field
name. Neither small network learns the teacher's strong clutter rejection.

Both student formats retain the existing architecture: FP32 35,204 bytes,
INT8 15,656 bytes. Distillation adds no runtime teacher or additional student
layers. Unchanged architecture is not a fresh speed measurement; no FPS claim
is made, especially for the original ARMv6 Pi Zero W.

## Evidence and reproduction

- `expanded-distillation-indoor-scenes.json`: teachers and seed-42 students,
  FP32/INT8, full predictions and hashes.
- `expanded-distillation-seed43-scenes.json`: matched repeat seed, both exports.
- `expanded-teacher-original-scenes.json`: teachers on original scenes/clutter.
- `expanded-distillation-original-scenes.json` and
  `expanded-distillation-seed43-original-scenes.json`: student regression checks.
- `expanded-distillation-errors.json`: seed-42 miss attribution.

Teacher output: `runs/balloon-red-blue-teacher-expanded-20261008`.
Student outputs: `runs/balloon-red-blue-expanded-distilled-20261008`,
`runs/balloon-red-blue-expanded-hardlabel-control-20261008`, and corresponding
`-seed43-20261008` runs. Artifacts and datasets remain ignored.

Teacher command (use a fresh output path when reproducing):

```sh
TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=1 .venv/bin/dtr --cpu train-teacher \
  --config configs/balloon-red-blue.json \
  --manifest data/balloon-red-blue-indoor-proposal-admit-20261008/manifest.json \
  --initial-teacher runs/balloon-red-blue-teacher-20261008/teacher.keras \
  --output runs/balloon-red-blue-teacher-expanded-20261008
```

Student command: `PYTHONPATH=.:src .venv/bin/python scripts/distill_pi_student.py`
with the same config/manifest, `--teacher` pointing to the expanded teacher,
fresh `--output`, `--student-variant separable_context --epochs 25
--learning-rate 0.001 --seed 42 --alpha 0.5`. Repeat with alpha 1, then seed 43
for both. No augmentation or initialization flag. For seed-43 distillation,
`--target-cache-run runs/balloon-red-blue-expanded-distilled-20261008` reuses
verified logits without changing student initialization or row order.

Export FP32 via `scripts/export_pi_float.py --run ... --output ...`.
Both scene evaluators now accept `.keras` teachers and `.tflite` students.
Teachers use bounded offboard batches (maximum 64); student inference remains
one crop at a time. Reports identify runtime and helper-source hash. This does
not alter Pi inference.

## Next bounded work

Prioritize training-only blue proposal coverage and transferring clutter
rejection without the recall collapse observed in earlier hard-negative runs.
An augmentation-consistent distillation experiment is reasonable, but cached
logits must match the exact augmented teacher input; do not reuse unaugmented
targets for transformed crops. Preserve both development panels, IMG exclusion,
and the final reserved test. Representative live Pi recordings and independent
field evidence remain required before any flight-readiness claim.

Verification: 371 tests passed, two skipped, six TensorFlow warnings; scoped
Ruff passed. No camera, SSH, ESP32, network or actuator changes in this work.
