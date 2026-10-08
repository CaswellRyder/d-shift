# Fresh student training and capacity check — 2026-10-08

Research only; no deployment promotion or 94% qualification claim. Three new
student runs completed. No teacher retraining, camera change, Pi installation,
ESP32 command or motor command was performed.

## Why this experiment

The existing proposal/geometric refinements warm-started a previous student and
still missed many targets. Compare fresh initialization of the existing 15,667
parameter context network and 7,763 parameter separable network, without changing
the data or operating threshold. Repeat the separable network with a second seed.
This tests training strategy/capacity, not a claim that a bigger backbone is better.

All runs use 25 epochs, Adam learning rate 0.001, batch size 16, 64x64 crops,
training-only right-angle rotations/reflections, and alpha=1 hard labels. Despite
the training script's name, these runs do NOT use teacher-logit distillation.
The frozen teacher is identity-checked but targets are marked unused zeros.
The original eight-crop validation hard loss selects the exported checkpoint;
the twelve indoor scenes do not select epochs. They are repeatedly used
development evidence, not final qualification.

Training manifest: `data/balloon-red-blue-indoor-proposal-admit-20261008/manifest.json`
SHA-256 `2aeadeb64c5b484e9e6f242af53e4911f615768adc84309b251cc2d691e218fb`.
510 training entries: 198 background, 174 red and 138 blue. Repeated/proposal views
are not independent scenes. Verified zero training entries with an `IMG_` source
basename. Calibration uses training data; the entire EngDes2 IMG family remains
evaluation-only and the original reserved test stays unevaluated.

Local runs (weights remain ignored):

- `runs/balloon-red-blue-capacity-context-20261008`: seed 42.
- `runs/balloon-red-blue-capacity-separable-20261008`: seed 42.
- `runs/balloon-red-blue-capacity-separable-seed43-20261008`: seed 43.

The checkout already contained uncommitted proposal-builder and augmentation
work on entry. It was preserved, not claimed or committed with this experiment.
The training script used has SHA-256
`51ab5e57c819a9f84a4d7b9dc5fe5cfab194d013b945a2be046ece7f145af10a`;
each local run records that identity and its full configuration/provenance.

## Twelve indoor development scenes

Same-class one-to-one matching, IoU >=0.5, threshold 0.8. These scenes contain
11 red and 12 blue targets and include negative frames. Under research-only MSER
with confirmed-parent suppression, FP32 results are:

- Prior geometric refinement: red TP/FP/FN 8/2/3, blue 7/4/5.
- Fresh context seed 42: red 9/2/2, blue 8/2/4.
- Fresh separable seed 42: red 11/2/0, blue 8/2/4.
- Fresh separable seed 43: red 9/1/2, blue 8/3/4.

The promising separable seed-42 result has red precision 84.6%, recall 100%;
blue precision 80%, recall 66.7%. The second seed shows meaningful variability;
do not present the better seed alone as a stable or independent accuracy result.
Two seeds are insufficient to establish a variance estimate.

With baseline search, fresh separable seed 42 gives red 9/1/2 and blue 7/0/5.
This further illustrates the search/precision tradeoff. Neither search meets the
target. Larger context did not improve the combined detection result enough to
justify promotion or assuming its extra compute is beneficial.

INT8 differs from FP32: context seed 42 gives MSER red 10/2/1, blue 8/2/4;
separable seed 42 gives red 9/2/2, blue 8/3/4. Seed-43 INT8 matches its FP32
aggregate counts here. Export-specific evaluation remains required.

On the original four development photos, both seed-42 FP32 networks give MSER
red 2/2/0 and blue 3/0/0. The persistent red false positives are not resolved.

## What the misses tell us

`scripts/audit_balloon_scene_errors.py` distinguishes proposal failure,
misclassification, below-threshold correct classifications, suppression and
one-to-one matching conflicts from saved predictions. It does not alter scores
or mine development examples into training.

For fresh separable seed 42 with MSER, the four blue misses split into:

- Two with no proposal box reaching IoU 0.5.
- Two with localized proposals but no correct blue classification.

None of these four is simply a correct-class score below threshold. Lowering a
global threshold is therefore not a complete remedy. Next work should improve
blue proposal coverage using training-only evidence and expand blue positive /
clutter discrimination without teaching from IMG evaluation images.

## Evidence and reproduction

- `capacity-indoor-scenes.json`: both architectures, seed 42, FP32/INT8.
- `capacity-original-scenes.json`: seed-42 FP32 on original development photos.
- `capacity-seed43-indoor-scenes.json`: repeat seed, FP32/INT8.
- `capacity-error-attribution.json`: per-target reasons, source receipt hash.

Reports retain model hashes, all detections and review hashes. No new Pi timing
was measured for these candidates; architecture size is not measured throughput.

Use `PYTHONPATH=.:src .venv/bin/python scripts/distill_pi_student.py` with the
red-blue config, manifest above, frozen red-blue teacher, fresh output directory,
`--student-variant context` or `separable_context`, `--alpha 1 --epochs 25
--learning-rate 0.001 --augmentation balloon-geometric --seed 42` (43 for repeat),
and no initialization/cache argument. Export FP32 with `scripts/export_pi_float.py`.
Evaluate using `scripts/evaluate_reviewed_balloon_scenes.py`, queue
`data/engdes2-development-scenes-20261008`, and the existing frozen review in
`configs/engdes2-development-scenes-20261008.json`. Keep thresholds unchanged.
