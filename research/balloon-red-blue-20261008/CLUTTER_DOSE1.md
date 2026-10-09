# Lower clutter dose: mixed, seed-sensitive results

2026-10-08. Four fresh training runs, same Pi-student architecture. No promotion.

## Result

Showing the eleven reviewed clutter crops once instead of four times recovers
recall in one seed, but worsens blue precision. A second seed does not reproduce
the recall gain. It does not solve the precision/recall tradeoff or reach the
94% target. Pi deployment, confidence threshold and runtime defaults remain
unchanged. No device FPS or live-camera accuracy was measured.

## Matched experiment

`scripts/build_clutter_ablation.py` adds a configurable bounded dose to the
existing 510-entry expanded indoor/proposal manifest. The intervention adds
eleven reviewed hard-negative crops once each; the matched control adds eleven
existing background entries. Both arms therefore have 521 training entries:
209 background, 174 red and 138 blue. Counts include repeated/correlated views,
not 521 independent photographs.

- Intervention manifest: `data/balloon-red-blue-clutter-dose1-20261008/manifest.json`,
  SHA-256 `faba59f034050b87a22c1066ca96b00bd1efc34dd58c225a28d335ec2f9de72f`.
- Control manifest: `data/balloon-red-blue-clutter-dose1-control-20261008/manifest.json`,
  SHA-256 `c1a358dd8bf4b277c3a32a3cdb20024c6eeb283474dd93122421978e2dbd35d4`.
- Parent manifest SHA-256:
  `2aeadeb64c5b484e9e6f242af53e4911f615768adc84309b251cc2d691e218fb`.
- Teacher: expanded MobileNetV4 Conv Small, SHA-256
  `db57446ecfba29c29353924792908d9a762f5452f50664f66c43cfdcc0ff45fe`.

The builder verifies the original review, permitted training sources and hashes,
rejects conflicts with expanded positives/holdout sessions, and rejects IMG-family
training sources. It refuses an existing output or a base with previously added
clutter. Validation/test metadata is preserved (only the relative path gains a
prefix). Test pixels are not materialized, read for training or evaluated.
Calibration is training-only; the entire EngDes2 IMG family stays evaluation-only.

Training uses fresh 7,763-parameter separable-context students, 64x64 RGB,
25 epochs, Adam 0.001, batch 16, alpha 0.5, temperature 4, no additional
geometric augmentation. Seeds 42 and 43 run in both arms. Original eight-crop
validation hard loss selects checkpoints. Each seed-43 run reuses only its own
arm's verified train teacher-logit cache, with student RNG reset to seed 43.

The new builder is separate from the pre-existing dirty indoor/proposal builder;
those changes remain preserved and uncommitted by this work. Its reviewed-negative
guard SHA-256 is recorded in the dataset receipt. The student training script
used remains `51ab5e57c819a9f84a4d7b9dc5fe5cfab194d013b945a2be046ece7f145af10a`.

## Full-scene development results

Twelve reviewed indoor scenes: 11 red and 12 blue targets, three negative frames.
Threshold 0.8, same-class one-to-one IoU >=0.5, ordinary MSER with confirmed-parent
suppression, twelve-candidate limit. FP32 TP/FP/FN:

- Seed 42 control: red 11/2/0; blue 8/3/4.
- Seed 42 one-dose clutter: red 11/1/0; blue 10/5/2.
- Seed 43 control: red 11/2/0; blue 8/3/4.
- Seed 43 one-dose clutter: red 10/2/1; blue 8/4/4.

Seed 42 reaches red precision/recall 91.7%/100% and blue 66.7%/83.3%.
Its two remaining blue misses have no sufficiently localized proposal; this
checkpoint correctly detects all localized targets but accepts too much clutter.
Seed 43 reaches red 83.3%/90.9% and blue 66.7%/66.7%, worse than its control.
Do not present the better seed alone as stable performance.

Relative to the earlier four-dose intervention, seed-42 red recall improves
9/11 -> 11/11 and blue 8/12 -> 10/12, but total false positives rise 0 -> 6.
Dataset size and optimizer update count also differ across those experiments;
the equal-size one-dose control is the appropriate within-experiment comparison.

INT8 seed-42 intervention loses one red detection versus FP32 (below threshold).
Its blue counts are unchanged. Seed-43 and both controls retain their FP32 MSER
aggregate counts. FP32 remains 35,204 bytes; INT8 15,656 bytes. No extra runtime
layers are added, but this is not proof of identical end-to-end Pi latency.

On the original four development photos with the same suppression, all four
FP32 models give red 2/2/0 and blue 3/0/0. On the eleven clutter crops, accepted
false detections drop 9 -> 5 for seed 42 and 11 -> 8 for seed 43. Those are
training-fit measurements for the interventions, not held-out accuracy.

Both scene panels are repeatedly used development evidence; the indoor panel
shares environments with training and has no verified recording-session
independence. Labels are AI-reviewed, not certified. Final test remains untouched.

## Evidence and reproduction

Ignored runs under `runs/` have the dataset names above, with additional
`clutter-dose1-seed43-20261008` and `clutter-dose1-control-seed43-20261008` suffixes
after the `balloon-red-blue-` prefix. Each retains provenance, logit cache,
epoch history, selected/final checkpoints and both exports.

Build with `PYTHONPATH=.:src .venv/bin/python scripts/build_clutter_ablation.py`,
the expanded proposal-admit manifest as `--manifest`, original bootstrap
manifest as `--original-manifest`, existing MSER negative `--queue`/`--review`,
fresh `--output`, `--mode admit` or `control`, and `--repetitions 1`.

Train via `scripts/distill_pi_student.py` using `configs/balloon-red-blue.json`,
each manifest, expanded teacher, fresh output and
`--student-variant separable_context --alpha 0.5 --epochs 25 --learning-rate 0.001
--seed 42` (then 43). No initialization or augmentation flags. Export FP32 with
`scripts/export_pi_float.py`; evaluate with the existing frozen scene reviews.

Receipts:

- `clutter-dose1-indoor-scenes.json`, `clutter-dose1-seed43-indoor-scenes.json`
- `clutter-dose1-original-scenes.json`, `clutter-dose1-seed43-original-scenes.json`
- `clutter-dose1-errors.json`

Next priority: broaden reviewed training-only balloon/clutter examples, especially
proposal-shaped blue targets, rather than continue dose tuning on these same few
scenes. Preserve source-family separation and distinguish search misses from
classification errors. Independent representative Pi-camera footage is still
needed for qualification; public-data experiments do not establish flight readiness.

Verification: 382 tests passed, two skipped, six known TensorFlow warnings;
scoped Ruff passed. Ten new tests cover dose bounds, protected test pixels,
IMG exclusion, positive/holdout conflicts and refusal to overwrite outputs or
admit clutter twice. No SSH, camera, network or ESP32/motor commands were issued.
