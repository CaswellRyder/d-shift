# Teacher-aligned crop augmentation and nested-box replay

2026-10-08. Four fresh small-student runs, paired controls, two seeds.
No model promotion, deployment changes, Pi measurement or flight qualification.

## Verdict

Fixed whole-crop geometric augmentation with recomputed teacher logits does not
improve these full-scene development results. Both augmented seeds lose one red
target relative to their same-size control; blue is worse in one seed and
unchanged in the other. Preserve this negative result rather than promote an
eight-crop validation score.

Miss attribution revealed a separate post-processing problem: a lower-confidence
large box can suppress a much stronger correctly localized balloon box inside it.
A confidence-ordered replay recovers that red target and removes a red false
positive in both augmented seeds, but regresses blue on a previous model. It is
research-only, not a universal fix. Neither experiment reaches 94% precision and
recall for both colors, and these small reused panels cannot qualify flight anyway.

## Teacher-aligned views

`scripts/build_paired_balloon_augmentation.py` adds one fixed non-identity D4
symmetry (right-angle rotation/reflection) for every training entry. All pixels,
color channels and the entire existing crop context are preserved. Rectangular
crops remain complete, swapping dimensions for odd quarter-turns. No new
backgrounds, cropped-away objects, new labels or synthetic scene generation.
This tests orientation variation of reviewed source/proposal-context crops,
not bounding-box jitter, added surrounding context or online random augmentation.

Each transformed view is materialized once as a PNG. The frozen teacher computes
new logits from that same file at 96x96; the student loads it at 64x64 using the
same stretch-to-square bilinear policy. The existing trainer's augmentation flag
stays `none`; its provenance describes no online augmentation, while the dataset
manifest records the offline transforms. An untransformed parent cache is rejected
by the changed manifest identity. Seed 43 reuses only its own arm's exact cache.

The matched control adds the original entry again. Both arms have 1,168 training
entries: 426 background, 410 red and 332 blue. The augmentation seed is fixed at
42 for both student seeds. There are 1,001 unique crop hashes in augmentation and
435 in control; these are not independent photographs or recording sessions.
The parent already contains repeats and correlated views.

Boundaries and guards:

- Require explicitly approved real balloon training data, reject repeated
  application, conflicting crop labels, source/holdout collisions and IMG sources.
- Preserve the eight validation and 24 reserved-test crop records, with only a
  relative path prefix change. Do not copy/open reserved-test pixels.
- Train and calibrate on training records only. The EngDes2 IMG family remains
  development-evaluation-only, not training, negative mining or calibration.
- Retain an unapproved manifest until output validation passes; refuse overwrites.
- Preserve all existing dirty trainer/proposal work without editing or committing it.

Dataset identities:

- Parent `data/balloon-red-blue-expanded-views-20261008/manifest.json`:
  `6011ae406f036759d9e6e4b334b5dd73de82b61248160d471c409ffede7e4cbb`.
- Augmentation `data/balloon-red-blue-paired-augmentation-20261008/manifest.json`:
  `99f07b308df5de072217810f63a2e9f98e14f50e082ad80005420915e7762f1d`.
- Control `data/balloon-red-blue-paired-augmentation-control-20261008/manifest.json`:
  `7b791391fc149a76ba0969fa639bd4aa9a2c7a40e4313278cab7d52f1c66bb01`.
- Builder SHA-256:
  `6002f07ad27bb1c0611dc0d8b1297b98763938d61d593386ddad329128b34fd7`.

The newly built manifests replace the stale bootstrap-only qualification wording
with explicit inherited-background and offline-augmentation descriptions. Existing
datasets, metadata and frozen artifact hashes are not rewritten.

## Training and full-scene results

Frozen expanded MobileNetV4 Conv Small teacher:
`runs/balloon-red-blue-teacher-expanded-20261008/teacher.keras`, SHA-256
`db57446ecfba29c29353924792908d9a762f5452f50664f66c43cfdcc0ff45fe`.

Students remain separable-context, 7,763 parameters, 64x64 RGB. Fresh initialization
seeds 42 and 43; Adam 0.001, batch 16, 25 epochs, alpha 0.5, temperature four.
Eight-crop validation hard-label loss selects checkpoints, not validation KL loss
against padded unused logits. Both arms have the same row counts and optimizer
updates. This is a matched comparison, not a comparison against a smaller dataset
with fewer updates per epoch. No teacher retraining or architecture changes.

Twelve indoor development scenes contain 11 red and 12 blue targets, including
three negative frames. Threshold 0.8, same-class one-to-one IoU >=0.5, 320x240 scan,
ordinary MSER search with a 12-candidate cap and confirmed-parent suppression.
FP32 TP/FP/FN:

- Seed 42 control: red 11/2/0; blue 9/2/3.
- Seed 42 augmented: red 10/3/1; blue 8/3/4.
- Seed 43 control: red 11/1/0; blue 7/4/5.
- Seed 43 augmented: red 10/2/1; blue 7/4/5.

INT8 reproduces these aggregate counts except seed-42 control, which changes red
to 10/3/1. That particular difference involves suppression, not a new missing
proposal. Scores can change post-processing decisions after quantization.
FP32 artifacts remain 35,204 bytes; INT8 15,656 bytes. Training augmentation adds
no deployed layers or preprocessing, but no new on-Pi latency measurement exists.

On the original four-scene development panel with the same suppression:
both seed-42 students give red 2/2/0 and blue 3/0/0; both seed-43 students give
red 2/2/0 and blue 3/1/0. On the eleven previously reviewed clutter-crop diagnostic,
false targets accepted are 10 augmented versus 9 control for seed 42, and 9 versus
9 for seed 43. These are research diagnostics, not independent accuracy.

Teacher-target audit on the actual cached TRAIN logits (argmax versus reviewed
hard labels): original exposures are background 212/213, red 204/205, blue 166/166
correct. Augmented counterparts are 210/213, 202/205 and 164/166. Eight teacher
predictions change across 584 paired exposures; control has zero changes and
identical counts in both halves. Geometry consistency is established, not teacher
invariance or label-perfect supervision. These are training-fit counts and do not
prove why student development performance regressed.

## Post-hoc containment experiment

`scripts/compare_balloon_containment.py` operates only on saved development
predictions. It restores only rows explicitly suppressed as confirmed balloon
parts; ordinary NMS, rejected background, thresholds and neural predictions stay
unchanged. It checks that replaying the old rule exactly reproduces the original
accepted flags and metrics before comparing anything.

The alternative processes accepted same-color boxes by descending classifier
confidence. If two boxes meet the existing containment criteria (smaller area
<=35% of larger; overlap >=95% of smaller), the stronger box wins. Exact score ties
prefer the larger box, preserving the original preference at ties. Suppressed
boxes cannot themselves suppress other boxes. This rule is general, with no
image IDs, target labels or ground-truth boxes used by the selection function.

New augmented students on the indoor panel, FP32 and INT8:

- Seed 42 red: 10/3/1 -> 11/2/0; blue unchanged at 8/3/4.
- Seed 43 red: 10/2/1 -> 11/1/0; blue unchanged at 7/4/5.
- Seed-42 control INT8 red: 10/3/1 -> 11/2/0; other control counts unchanged.
- The original four-scene panel's counts remain unchanged for all four FP32 runs.

Checks on earlier saved models prevent interpreting this as a universally safe
fix. The previous expanded teacher's counts stay unchanged. Both later teacher
updates recover one red target and remove one red false positive (9/1/2 -> 10/0/1).
However, the earlier fixed-teacher/new-views seed-42 student's blue result worsens
from 9/4/3 to 8/5/4, in both FP32 and INT8. Its seed-43 red result improves
10/2/1 -> 11/1/0 with blue unchanged. Confidence is not a reliable localization
quality estimate: a confident fragment can beat the full balloon just as an
oversized parent can hide a better small box. Do not enable this unconditionally.

This hypothesis was selected after examining development failures, not preregistered
or independently tested. It adds no neural evaluations to the saved-replay
comparison, but does not demonstrate actual device speed. Neither search nor
production runtime modules were edited; no alternate rule was deployed.

## Evidence and reproduction

Build both datasets with:

```sh
PYTHONPATH=.:src .venv/bin/python scripts/build_paired_balloon_augmentation.py \
  --manifest data/balloon-red-blue-expanded-views-20261008/manifest.json \
  --output data/balloon-red-blue-paired-augmentation-20261008
```

Repeat with `--control` and the control output above. Use fresh paths on reruns.
Train with `PYTHONPATH=.:src .venv/bin/python scripts/distill_pi_student.py`,
`--config configs/balloon-red-blue.json`, each manifest, the frozen teacher,
fresh `--output`, and `--student-variant separable_context --alpha 0.5 --epochs 25
--learning-rate 0.001 --seed 42` (then 43). Do not pass the online augmentation flag.
Use `--target-cache-run` only for the exact arm's seed-42 output. Export FP32 with
`scripts/export_pi_float.py`; INT8 is exported by training. Run prefix:
`runs/balloon-red-blue-paired-augmentation`, with suffixes `-20261008`,
`-control-20261008`, `-seed43-20261008`, and `-control-seed43-20261008`.
Trainer SHA-256 remains
`51ab5e57c819a9f84a4d7b9dc5fe5cfab194d013b945a2be046ece7f145af10a`.

Evaluate using the unchanged indoor and original-development review configs with
`scripts/evaluate_reviewed_balloon_scenes.py` and
`scripts/evaluate_red_blue_development.py`. Run error attribution with
`scripts/audit_balloon_scene_errors.py`. Reports in this directory use
`paired-augmentation` and `paired-augmentation-seed43` prefixes, each with
`-indoor-scenes.json`, `-indoor-errors.json` and `-original-scenes.json`.
Per-detection evidence, model hashes and evaluation code identities are retained.

Replay containment with:

```sh
PYTHONPATH=.:src .venv/bin/python scripts/compare_balloon_containment.py \
  --report research/balloon-red-blue-20261008/paired-augmentation-indoor-scenes.json \
  --output research/balloon-red-blue-20261008/paired-augmentation-indoor-containment.json
```

Repeat on both seeds' indoor/original reports and existing
`new-views-teachers-indoor-scenes.json`, `new-views-fixed-teacher-indoor-scenes.json`,
and `new-views-fixed-teacher-seed43-indoor-scenes.json`. Derived receipts replace
the `-scenes.json` suffix with `-containment.json` and bind the source report hash.

## Next bounded work and qualification boundary

Prioritize a training-only proposal-localization quality signal rather than more
orientation sweeps or score-threshold tuning. It should distinguish whole-balloon
boxes from fragments and multi-balloon parent regions. Use approved training
annotations/reviews, preserve both development panels, and do not turn IMG misses
into mined training examples. Evaluate alongside the fixed stronger teacher and
the existing small student; measure any runtime additions on the actual Pi before
claiming a speed/accuracy improvement.

Two blue targets still lack sufficiently localized proposals even for the stronger
teacher; student retraining alone cannot recover those with unchanged proposals.
The current eight-crop checkpoint selector is too small for robust selection.
Broader independently assigned development data and representative Pi footage
remain necessary. Current public-data scenes share environments with training,
are repeatedly inspected, and have AI-reviewed rather than certified labels.
The reserved final test was not evaluated. ESP32 and motors remain untouched.

Verification: 19 augmentation guard tests and 14 containment replay tests added;
429 tests passed, two skipped, six known TensorFlow warnings. Scoped Ruff passed.
Augmentation builder/test checkpoint: `adff1743`. Only owned paths are committed;
data, weights and teacher-target caches remain ignored. No SSH, camera, network,
ESP32 or motor commands were issued during this work.
