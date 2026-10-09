# Additional reviewed training views

2026-10-08. Training-only public-data expansion, paired controls and two student
seeds. No Pi deployment or flight-qualification claim.

## Dataset and review

Prepared 22 additional frames from the same two EngDes2 filename families already
assigned to training. These are additional views, not verified independent
recording sessions. The source includes flips, blur and noise augmentations.
Prior source groups and flip-aware perceptual near-duplicates of earlier training
frames are excluded. The pre-existing holdout duplicate screen reads held-out
pixels for perceptual hashing only; no held-out predictions, labels for fitting,
or model selection were used. The entire IMG filename family remains excluded
from training, calibration and negative mining.

AI visual review covered all 22 full frames and all 77 proposed crops. Explicitly
admitted 74: 31 red, 28 blue, 15 background. Excluded IDs 43, 47 and 75. In
particular, 47 was proposed as background but visibly contains a clipped red
balloon; absence from the public annotation is not evidence of background.
The ambiguous food-packaging region and blurred cafe patch were also excluded.
Review is not human-certified and does not establish independent evaluation.

`scripts/prepare_indoor_balloon_review.py` now supports bounded `--per-family`
selection and a verified `--exclude-queue`. The expansion path uses research
MSER proposals for review; the original default review behavior stays unchanged.
`scripts/append_reviewed_balloon_views.py` requires an explicit complete review,
verifies the training base and source boundaries, rejects held-out conflicts,
and refuses overwrites. The control substitutes existing crops of the same class
using fixed RNG 42. Both arms therefore contain 584 training entries:
213 background, 205 red and 166 blue, compared with 510 parent entries. Repeated
entries count toward optimizer exposure, not independent examples.

The original eight validation and 24 reserved-test crop records are preserved.
Test pixels are not copied to the new datasets. Training and calibration use
training-only records; the test split was not evaluated.

Receipts and identities:

- Queue: `data/engdes2-indoor-expansion-review-20261008/review.json`, SHA-256
  `f6a7231360ea4a586d25120adfedc0c2ca9f1aeccb23e8e855289b27e1d1a5f6`.
- Review: `configs/engdes2-indoor-expansion-review-20261008.json`.
- Parent: `data/balloon-red-blue-indoor-proposal-admit-20261008/manifest.json`,
  SHA-256 `2aeadeb64c5b484e9e6f242af53e4911f615768adc84309b251cc2d691e218fb`.
- Intervention: `data/balloon-red-blue-expanded-views-20261008/manifest.json`,
  SHA-256 `6011ae406f036759d9e6e4b334b5dd73de82b61248160d471c409ffede7e4cbb`.
- Control: `data/balloon-red-blue-expanded-views-control-20261008/manifest.json`,
  SHA-256 `1e4349eaaa5baa6ec8064a4908f015d18d275fa56489239c01a9515332f1abac`.

The inherited `dataset_qualification.negatives` text still describes the original
bootstrap and incorrectly says non-balloon red/blue clutter is missing. This is
stale descriptive metadata, not the actual sample inventory: the review and
manifest receipts above identify the added clutter. Existing hashed artifacts
are retained unchanged; correct this text in a future versioned dataset build.

## Experiment A: update teacher and student together

Both MobileNetV4 Conv Small teachers warm-start from the previous expanded
teacher (`db57446ecfba29c29353924792908d9a762f5452f50664f66c43cfdcc0ff45fe`).
The intervention uses new views; its matched control repeats existing crops.
Both use 96x96 RGB, seed 42, five head epochs and up to 15 fine-tuning epochs,
patience five, with the eight-crop validation loss selecting checkpoints.

Intervention selects fine-tuning epoch two, validation loss 0.002705; control
selects epoch five, loss 0.000615. Both classify all eight validation crops
correctly, but that does not imply good full-scene detection.

Teacher SHA-256 identities:

- New views: `d96fbcc4a28dd970211ec2e70e2fae56126dddf68416b97e22739c637eb1c77c`.
- Matched control: `e7863f88059ed108340f5d9e571e6a93ee056ca17a4847314be39db4275378bc`.

Each corresponding teacher distills fresh 7,763-parameter separable-context
students: 64x64 RGB, Adam 0.001, batch 16, 25 epochs, alpha 0.5, temperature four,
no additional geometric augmentation. Seeds 42 and 43 are paired across arms.
Selection uses validation hard-label loss; seed 43 reuses only its own arm's
verified teacher-target cache and resets student RNG. Teacher seed is not varied.
This measures the combined teacher/student data intervention, not a student-only
or teacher-only causal effect.

### Twelve-scene development panel

11 red and 12 blue targets, three negative frames, 320x240 images. Threshold 0.8,
same-class one-to-one IoU >=0.5, ordinary MSER with confirmed-parent suppression.
Counts below are TP/FP/FN; all students are FP32 unless noted.

- Previous teacher: red 11/1/0; blue 10/1/2.
- New-views teacher: red 9/1/2; blue 10/1/2.
- Control teacher: red 9/1/2; blue 10/1/2.
- Seed 42 new-views student: red 9/0/2; blue 8/2/4.
- Seed 42 control student: red 10/1/1; blue 8/2/4.
- Seed 43 new-views student: red 10/2/1; blue 8/3/4.
- Seed 43 control student: red 11/2/0; blue 7/4/5.

Both teacher updates lose two red detections versus the parent. New views lose
one red detection versus control in each student seed; seed 43 gains one blue
detection and removes one blue false positive. This is a tradeoff, not a robust
improvement. INT8 matches these counts except seed-42 intervention, which loses
another red detection (8/0/3). Quantization is not automatically accuracy-neutral.

For the new-views FP32 students, both seeds miss two blue targets with no
sufficiently localized proposal and two through wrong classification. Seed 42
misses two red targets below confidence threshold; seed 43 has one red miss
involving suppression. Error attribution does not change thresholds or search.

### Original four-scene development panel

Teachers: red 2/0/0 and blue 3/0/0 in both arms with confirmed-parent suppression.
Seed-42 students both produce red 2/2/0, blue 3/0/0. Seed-43 intervention produces
red 2/1/0, blue 3/0/0 versus control red 2/2/0, blue 3/1/0.
On eleven previously reviewed clutter crops, intervention students accept six
false targets and controls nine in both seeds. Teachers accept zero versus one.
These are repeatedly inspected research diagnostics, not independent accuracy;
the evaluator's legacy `reviewed_negative_training_fit` field name does not mean
these eleven crops were appended in this particular experiment.

## Experiment B: keep the stronger teacher fixed

Because both updated teachers regressed on red recall, a second paired experiment
uses the unchanged previous expanded teacher in both arms. Dataset, student
architecture, training settings, checkpoint selection and seeds otherwise match
experiment A. This isolates the data change during student training from teacher
retraining. The hypothesis was chosen after A, so this remains exploratory work.

Indoor confirmed-parent MSER FP32 TP/FP/FN:

- Seed 42 new views: red 11/0/0; blue 9/4/3.
- Seed 42 control: red 10/3/1; blue 8/3/4.
- Seed 43 new views: red 10/2/1; blue 8/4/4.
- Seed 43 control: red 11/2/0; blue 8/3/4.

INT8 preserves the aggregate counts in these four comparisons. Seed 42's perfect
red detection is only eleven targets in a repeatedly inspected development panel,
not red-balloon flight qualification. Blue precision/recall is 69.2%/75.0% in
seed 42 and 66.7%/66.7% in seed 43. The second seed loses red recall and worsens
blue precision versus its control. The extra views still do not establish a
repeatable improvement. Do not select seed 42 and hide seed 43.

On the original four scenes, both seed-42 students give red 2/2/0 and blue 3/0/0.
Seed-43 new views give red 2/2/0, blue 3/1/0 versus control red 2/2/0, blue 3/0/0.
On the eleven clutter-crop diagnostic, accepted false targets are 6 versus 10
in seed 42, but 9 versus 8 in seed 43. These results reinforce the instability.

Reproduce with the same distillation/export commands but `--teacher` fixed to
`runs/balloon-red-blue-teacher-expanded-20261008/teacher.keras` for both arms.
Run prefix: `runs/balloon-red-blue-new-views-fixed-teacher`, followed by
`-20261008`, `-control-20261008`, `-seed43-20261008`, and
`-control-seed43-20261008`. Cache reuse remains arm-specific.
Reports use `new-views-fixed-teacher` and `new-views-fixed-teacher-seed43` prefixes
with `-indoor-scenes.json`, `-indoor-errors.json` and `-original-scenes.json`.

## Decision and next work

Completed two teacher updates and eight small-student runs across two paired
experiments. Neither experiment establishes a stable gain or reaches 94% for
both colors. Keep deployment artifacts, runtime settings and thresholds unchanged.
All results are offboard development evidence; no new Pi FPS was measured.

The added data and guarded review workflow remain useful research assets. The
previous expanded teacher remains the stronger full-scene reference here. Next,
focus on the student transfer/search gap with that teacher frozen: distinguish
missed proposals from wrong classes, and test training-only proposal-context
augmentation with teacher targets recomputed for exactly the same transformed
input. Do not pair transformed inputs with stale untransformed target caches.
The eight-crop checkpoint selector is very small; a broader, separately assigned
development validation set is also needed, without recycling IMG or the reserved
final test into training or calibration. New-view count alone is not progress
toward an independently measured accuracy target.

The two development panels have been used repeatedly, share environments with
training, and do not have independently verified recording sessions. Ground
truth is AI-reviewed. Representative independent Pi-camera footage, live timing,
and end-to-end integration checks are still required for flight qualification.

## Reproduction and evidence

Use fresh output paths; all builders/exporters refuse overwrites. From repo root:

```sh
PYTHONPATH=.:src .venv/bin/python scripts/prepare_indoor_balloon_review.py \
  --root data/raw/engdes2-red-blue-v1-20261008 \
  --manifest data/balloon-red-blue-bootstrap-20261008/manifest.json \
  --output data/engdes2-indoor-expansion-review-20261008 \
  --per-family 12 \
  --exclude-queue data/engdes2-indoor-review-20261008/review.json

PYTHONPATH=.:src .venv/bin/python scripts/append_reviewed_balloon_views.py \
  --manifest data/balloon-red-blue-indoor-proposal-admit-20261008/manifest.json \
  --original-manifest data/balloon-red-blue-bootstrap-20261008/manifest.json \
  --queue data/engdes2-indoor-expansion-review-20261008 \
  --review configs/engdes2-indoor-expansion-review-20261008.json \
  --output data/balloon-red-blue-expanded-views-20261008
```

Repeat the builder with `--control` and the control output path. Review decisions
must match the verified queue, not be reused blindly on different inputs.

Teacher command: `.venv/bin/dtr --cpu train-teacher`, with
`--config configs/balloon-red-blue.json`, each `--manifest`,
`--initial-teacher runs/balloon-red-blue-teacher-expanded-20261008/teacher.keras`,
and outputs `runs/balloon-red-blue-teacher-new-views-20261008` and
`runs/balloon-red-blue-teacher-new-views-control-20261008` respectively.
Set `TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=1`.

Student command: `PYTHONPATH=.:src .venv/bin/python scripts/distill_pi_student.py`,
same config, each manifest/teacher pair, fresh output and
`--student-variant separable_context --alpha 0.5 --epochs 25
--learning-rate 0.001 --seed 42` (then 43). Seed-43 cache reuse uses
`--target-cache-run` pointing at the same arm's seed-42 run. Export FP32 via
`scripts/export_pi_float.py --run ... --output .../student.fp32.tflite`.
INT8 export is produced by training. Artifact sizes remain 35,204 bytes FP32 and
15,656 bytes INT8; no extra runtime layers or measured Pi speed gain.

Student runs use the prefix `runs/balloon-red-blue-new-views-student`, followed by
`-20261008`, `-control-20261008`, `-seed43-20261008` and
`-control-seed43-20261008`. Artifacts and datasets remain ignored.

Evaluate with `scripts/evaluate_reviewed_balloon_scenes.py` using
`data/engdes2-development-scenes-20261008` and its existing review config; then
`scripts/evaluate_red_blue_development.py` using the original bootstrap manifest,
original scene review and MSER negative review. Saved reports preserve individual
predictions, model hashes and evaluation-source hashes. Generate error attribution
with `scripts/audit_balloon_scene_errors.py --report ... --output ...`.

Receipts for experiment A:

- `new-views-teachers-indoor-scenes.json`, `new-views-teachers-indoor-errors.json`
- `new-views-students-indoor-scenes.json`, `new-views-students-indoor-errors.json`
- `new-views-original-scenes.json`
- `new-views-students-seed43-indoor-scenes.json`
- `new-views-students-seed43-indoor-errors.json`
- `new-views-students-seed43-original-scenes.json`

Code identities used: expansion builder
`ae400252a32a95d0bf26be91e90ebf9337fe45d51372f9b44ecb8bcb7e0eb9c3`,
pre-existing admission guard
`3e68c17bfc0a7c93c6dc8d25f8d664b4ab9bc80e8585a0424dffa0b6377d5c08`,
student trainer
`51ab5e57c819a9f84a4d7b9dc5fe5cfab194d013b945a2be046ece7f145af10a`.
Pre-existing uncommitted indoor/proposal/augmentation changes remain preserved.

Verification: 396 tests passed, two skipped, six known TensorFlow warnings;
scoped Ruff passed. Fourteen new tests cover selection bounds, previous queue
integrity, full review requirements, matching control counts, source boundaries,
test-pixel non-materialization and overwrite refusal. No SSH, camera, network,
ESP32 or motor changes were made.
