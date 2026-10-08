# Clutter training with the expanded positive corpus

2026-10-08. Four new student runs, two matched training seeds. Research only.

## Decision

The validation-selected clutter models reduce false positives in both seeds,
but also lose red recall. Fixed-final-epoch results are less consistent. Keep
this as a research alternative; do not replace the Pi model or claim the 94%
gate. The blue detection/search bottleneck remains.

No Pi, camera, network, ESP32 or actuator changes were made. No new device
timing was measured. Existing runtime/search defaults and frozen weights remain
unchanged; nothing was pushed remotely.

## Controlled design

Both arms start fresh, not from the earlier warm-start refinement. They use the
expanded reviewed indoor/proposal positive corpus and the same updated offboard
MobileNetV4 Conv Small teacher:
`runs/balloon-red-blue-teacher-expanded-20261008/teacher.keras`, SHA-256
`db57446ecfba29c29353924792908d9a762f5452f50664f66c43cfdcc0ff45fe`.

Both training manifests have 554 entries: 242 background, 174 red and 138 blue.
The experimental arm adds eleven reviewed hard-negative crops repeated four
times; its control substitutes 44 existing background entries. Repetitions and
proposal views are not independent images. No new labels were inferred from
unannotated regions. The hard negatives already have source-context review in
`configs/balloon-red-blue-mser-review-20261008.json`.

- Intervention manifest: `data/balloon-red-blue-balanced-clutter-20261008/manifest.json`,
  SHA-256 `600a28b90f489c342a0362c28b76d71b97cd999071a0bfd4cb30654e875c0c00`.
- Control manifest: `data/balloon-red-blue-balanced-clutter-control-20261008/manifest.json`,
  SHA-256 `c72fbaa7b0d828656f26577ea0bf4c7d20fa6279860c9b38000044216ade0e1f`.

Verified zero training source basenames in the reserved `IMG_` family in both
manifests. All original validation/test metadata is retained; test pixels are
not copied or opened. The IMG development family is not used for training,
calibration or negative mining. The original final test remains unevaluated.

Training: separable-context student, 64x64 RGB, 7,763 parameters, 25 epochs,
Adam 0.001, batch 16, no additional geometric augmentation. Distillation uses
alpha 0.5 and temperature 4. Both seed 42 and seed 43 are run in both arms.
The existing eight-crop validation hard loss selects checkpoints, not the
full-scene panel. Seed 43 reuses only its own arm's hash-bound train logits;
student initialization/shuffle is reset to seed 43.

The pre-existing dirty builder, proposal and augmentation edits were preserved
and used, not committed as this turn's work. Builder SHA-256:
`3e68c17bfc0a7c93c6dc8d25f8d664b4ab9bc80e8585a0424dffa0b6377d5c08`.
Student training script SHA-256:
`51ab5e57c819a9f84a4d7b9dc5fe5cfab194d013b945a2be046ece7f145af10a`.

## Twelve indoor development scenes

Threshold 0.8, same-class one-to-one IoU >=0.5, 320x240, at most twelve candidate
crops, ordinary MSER with confirmed-parent suppression. The scenes contain
11 red and 12 blue targets, including three negative frames. TP/FP/FN, FP32:

- Seed 42 control: red 11/2/0, blue 8/2/4.
- Seed 42 clutter: red 9/0/2, blue 8/0/4.
- Seed 43 control: red 10/3/1, blue 8/3/4.
- Seed 43 clutter: red 9/1/2, blue 8/1/4.

Each seed eliminates four false positives versus its matched control. Red
recall drops from 100% to 81.8% for seed 42 and from 90.9% to 81.8% for seed 43.
Blue recall remains 66.7%. Seed-42 clutter has zero false positives on this
small panel; seed 43 still has one per color. Do not report the seed-42 100%
precision alone as reliable field performance. Two seeds do not establish a
variance estimate or independent validation.

INT8 reproduces these MSER aggregate counts for both arms/seeds. It does not
always reproduce baseline-search results: seed-42 clutter INT8 recovers one red
target missed by FP32. Keep export-specific checks.

The seed-42 miss audit attributes its two missed reds to one misclassification
and one below-threshold correct classification. Four blue misses remain two
unlocalized proposals and two misclassifications. Lowering the global threshold
cannot solve all of these errors and was not attempted here.

## Regression and training-fit checks

On the original four development photos with the same MSER suppression, both
seeds give control red 2/2/0 and blue 3/0/0; clutter red 2/1/0 and blue 3/0/0.
One red false positive persists. Results are search-dependent: seed-43 control
has zero red false positives with baseline search, while its clutter arm has
one. The clutter arm is not uniformly better on every operating condition.

On the eleven added clutter crops, seed-42 accepted false detections decrease
11 -> 2, and seed-43 decreases 8 -> 2. This is training fit for the intervention,
not independent accuracy. On 78 admitted indoor training crops, both seeds of
the intervention accept 24/24 red and 23/24 blue correctly and reject all 30
backgrounds; controls accept 24/24 of each color and reject all backgrounds.
The earlier severe training-positive collapse is not reproduced, but some
development red recall is still lost.

All runs pass eight-crop argmax validation. That does not establish detection
quality or justify promotion. Indoor panels are repeatedly inspected,
seen-environment development data with AI-reviewed, not certified, labels;
recording-session independence is unverified. Neither panel is representative
live Pi-camera flight qualification.

## Artifacts and reproduction

### Fixed-final-epoch diagnostic

After observing the selected-checkpoint tradeoff, evaluated the already-trained
epoch-25 endpoint of all four runs. This is a post-hoc diagnostic, not an epoch
sweep or replacement selection rule. FP32 indoor MSER TP/FP/FN:

- Seed 42 final control: red 11/1/0, blue 9/3/3.
- Seed 42 final clutter: red 9/0/2, blue 8/1/4.
- Seed 43 final control: red 10/3/1, blue 8/3/4.
- Seed 43 final clutter: red 10/3/1, blue 8/5/4.

Longer training does not consistently solve the tradeoff. Seed 43 recovers one
red versus its selected clutter checkpoint but has more false positives and is
worse than its final control for blue precision. Seed 42 loses one blue target
relative to its final control as well as two reds. No endpoint is promoted.

On the original four scenes, final MSER seed-42 clutter/control both give red
2/2/0 and blue 3/0/0; seed-43 clutter/control give red 2/1/0 versus 2/2/0 and
blue 3/0/0 for both. All full predictions remain in the two `final` receipts.

### Local outputs

Ignored outputs under `runs/`:

- `balloon-red-blue-balanced-clutter-20261008`
- `balloon-red-blue-balanced-clutter-control-20261008`
- `balloon-red-blue-balanced-clutter-seed43-20261008`
- `balloon-red-blue-balanced-clutter-control-seed43-20261008`

Each holds provenance, epoch logs, best/final Keras checkpoints, FP32 and INT8
exports. FP32 stays 35,204 bytes and INT8 15,656 bytes. Runtime architecture is
unchanged; accepted detections/tracking load can differ, so do not infer equal
end-to-end Pi FPS from equal parameter count.

Use `scripts/build_indoor_balloon_training.py` with the original bootstrap
manifest, indoor queue/review, `--proposal-mode admit` and the reviewed indoor
proposal queue. Add the existing MSER hard-negative queue/review and choose
`--hard-negative-mode admit` or `control`; use fresh output paths.

Run `scripts/distill_pi_student.py` with each new manifest and expanded teacher,
fresh output, `--student-variant separable_context --alpha 0.5 --epochs 25
--learning-rate 0.001 --seed 42` (then 43). No warm-start or augmentation flag.
For seed 43, pass `--target-cache-run` for the matching seed-42 arm. Export FP32
via `scripts/export_pi_float.py`. Use the same frozen reviews with
`evaluate_reviewed_balloon_scenes.py` and `evaluate_red_blue_development.py`.

Committed receipts:

- `balanced-clutter-indoor-scenes.json`, `balanced-clutter-seed43-indoor-scenes.json`
- `balanced-clutter-original-scenes.json`, `balanced-clutter-seed43-original-scenes.json`
- `balanced-clutter-training-fit.json`, `balanced-clutter-seed43-training-fit.json`
- `balanced-clutter-errors.json`
- `balanced-clutter-final-indoor-scenes.json`, `balanced-clutter-final-original-scenes.json`

Next: test a lower reviewed-clutter weight or augmentation-consistent teacher
supervision with matched controls; preserve positive recall and both development
panels. Separate search localization from classification misses. Do not widen
search just to increase proposal coverage: `BLUE_RED_SEARCH.md` records another
coverage improvement that failed the full-scene false-positive check.

Verification: 372 tests passed, two skipped, six known TensorFlow warnings;
scoped Ruff passed. No flight readiness or speed improvement is established.
