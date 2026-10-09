# Tiny proposal-localization quality head

2026-10-08. New training supervision and two frozen-feature model extensions.
Research only: not deployed, not Pi-timed, not at the 94% flight-readiness target.

## Result

A 129-parameter auxiliary head can run alongside the existing small classifier
without changing its class predictions. Both exported models retain exactly the
previous class probabilities on all candidates in the two development panels.
The export grows from 35,204 to 35,972 bytes: 768 additional bytes, about 2.2%.
Total parameters rise from 7,763 to 7,892. No second backbone or teacher inference
is required at runtime; actual Pi latency remains unmeasured.

However, ordering nested boxes by predicted localization quality reproduces the
previous confidence-ordering tradeoff. It recovers a red detection in one seed
but worsens blue detection in the other. Do not promote these artifacts or change
the deployed suppression rule. Preserving classifier logits is necessary for
this experiment but is not sufficient for better full-scene detection.

## Training-only quality data

Prepared and visually reviewed 144 MSER proposal crops from 38 previously assigned
training frames: 52 proposals from the original 16-frame review, and 92 from the
22-frame expansion. Reviewed all four crop sheets and all five full-frame sheets.
All 144 proposals were admitted for quality regression after inspection. These
are additional labels/views of existing training images, not independent scenes.

The continuous target is the highest IoU of the candidate's **box** with one
reviewed visible-balloon annotation, not the union of multiple balloon boxes and
not class probability. The image input remains the candidate's padded crop.
Clipped edge balloons use their visible annotated boxes; this is localization to
the current annotation convention, not reconstruction of an invisible full object.

- 128 proposals overlap an admitted balloon annotation by at least 0.02 IoU.
- 16 are exact crops of already reviewed backgrounds; their quality target is zero.
- 40 targets are below 0.5 IoU and 104 are at least 0.5.
- Unmatched proposals without an exact reviewed negative crop are unknown and
  excluded automatically, never inferred to be background from missing labels.
- Oversized person/balloon, room/balloon and multi-balloon boxes receive low best-
  single-object IoU. Body fragments can also receive low quality even when their
  crop class is correctly red or blue. The quality head is class-agnostic.

Frames have incomplete public annotations. Review establishes only these admitted
quality targets against known anchors, not complete-frame truth. Labels are
AI-reviewed, not human-certified. Public flips/noise/blur and shared environments
remain correlated. Source/filename groups are not verified recording sessions.

`scripts/prepare_balloon_quality.py` reads only the train annotation member of the
hash-bound EngDes2 archive and previously reviewed train-frame pixels. It creates
a quarantined queue. Admission requires a matching explicit review for every
proposal. `scripts/train_balloon_quality.py` rechecks parent queue/review hashes,
original manifest identity, training source/family, anchor identity, crop/frame
checksums, contained paths, box geometry and recomputed quality targets. Exact
background targets must match a previously admitted background crop.

The entire IMG family remains excluded from training, calibration and negative
mining. The original reserved final test was not opened or evaluated. Existing
development panels are evaluation-only. No new dataset/model download was needed.

Source identities:

- Original quality queue: `data/balloon-quality-review-20261008/review.json`,
  SHA-256 `e5f964224d49ba610b88f99e6579c081b757a0b44d59136fbfce7f3ac65dd992`.
- Expansion quality queue: `data/balloon-quality-expansion-review-20261008/review.json`,
  SHA-256 `b23b665efda36800116240e02a9f74010888b6ea513903ee426f475ee817334d`.
- Reviews: `configs/balloon-quality-review-20261008.json` and
  `configs/balloon-quality-expansion-review-20261008.json`.
- Training config: `configs/balloon-quality-training-20261008.json`, SHA-256
  `cc9f64fc2f3f70cca4151fddc489743bb4276ca4e7513197b62d1d4a82877fe6`.

## Frozen-feature fit and export

Parents are the two new-views students distilled from the unchanged stronger
MobileNetV4 teacher: `runs/balloon-red-blue-new-views-fixed-teacher-20261008` and
`runs/balloon-red-blue-new-views-fixed-teacher-seed43-20261008`.
These are the original seed-42/43 student checkpoints, not newly randomized heads
or eight additional backbone training runs. Their full 64x64 RGB classifier stays
frozen. The source Keras identity is verified against the prior FP32 metadata,
whose TFLite hash is also checked.

Extract 128 existing `spatial_features`, standardize them using training means
and standard deviations, and fit a linear ridge regressor with fixed penalty 10.
The intercept is not penalized. Standardization is folded back into the 128
weights and one bias, so deployment needs only a Dense(1) branch. The original
three class logits are concatenated with the raw quality regression. Quality is
clipped to [0,1] by the research predictor; it is not passed through the class
softmax and is not advertised as a calibrated probability.

Five deterministic source-group folds provide a head-only diagnostic. Each fold
fits normalization and regression on its training groups only. The fixed penalty
is not swept against development scenes. The backbone has already learned from
related crops and environments, so this is not independently held-out model
accuracy or an honest unseen-environment qualification score.

- Seed 42: training quality MAE 0.1007; grouped head-only MAE 0.1479.
- Seed 43: training quality MAE 0.0924; grouped head-only MAE 0.1381.
- Keras classifier logits are bit-identical before/after head attachment on all
  144 training proposals.
- Maximum combined-output Keras/TFLite difference: 8.58e-6 and 4.77e-6 respectively.
- Class probability difference from saved original TFLite predictions is exactly
  zero on both the 12-scene indoor and four-photo original panels, for both seeds.

Artifacts are FP32 only; no INT8 quality-head calibration/export is claimed.
There is a dedicated four-output research predictor. The existing three-class
runtime correctly rejects these models with `Output classes do not match metadata`.
They cannot silently become the deployed classifier by pointing the old runtime
at the new file.

Artifact identities:

- `runs/balloon-quality-head-20261008/quality.fp32.tflite`:
  `3ecaee001eb948df0bcc38c76bb62e1f83b3f00f95f5f73d49f09fb4dd8f1b71`.
- `runs/balloon-quality-head-seed43-20261008/quality.fp32.tflite`:
  `7fba3d53b92727b31950a000ec98ee87857010715be5c2ebb1faa0f3cf6511e5`.
- Training script SHA-256:
  `909f0f6cc7766f344a261f484e5444aa3406acf322b880773a5b74c944cb92c0`.

## Full-scene comparison

`scripts/evaluate_balloon_quality.py` loads the hash-bound scene pixels and saved
MSER proposal boxes, computes new joint model outputs, and verifies each class
probability, label and raw confidence acceptance against the source report.
It preserves the 0.8 class threshold, ordinary NMS and candidate search/cap.
The previous parent-first suppression must reproduce its original aggregate
counts. Only containment ordering changes.

Quality ordering handles accepted same-color boxes by descending predicted IoU,
then confidence, then larger area for ties. Containment uses the existing 35%
area and 95% coverage criteria. This is not an additional quality rejection
threshold. Rejected background and ordinary-NMS boxes cannot be resurrected.
No frame IDs, reference boxes or target labels enter the selection rule.

Indoor panel: 11 red and 12 blue targets across 12 frames. Counts are TP/FP/FN.

- Seed 42 parent-first: red 11/0/0; blue 9/4/3.
- Seed 42 confidence-ordered: red 11/0/0; blue 8/5/4.
- Seed 42 quality-ordered: red 11/0/0; blue 8/5/4.
- Seed 43 parent-first: red 10/2/1; blue 8/4/4.
- Seed 43 confidence-ordered: red 11/1/0; blue 8/4/4.
- Seed 43 quality-ordered: red 11/1/0; blue 8/4/4.

The new quality ordering does not improve on simple confidence ordering here.
Seed 43's red precision/recall becomes 91.7%/100%, but blue remains 66.7%/66.7%.
Seed 42's blue worsens to 61.5%/66.7%. Perfect red on eleven repeatedly inspected
targets is not evidence of 94% flight-ready red detection.

On the original four photos all three ordering methods agree: seed 42 gives
red 2/2/0 and blue 3/0/0; seed 43 gives red 2/2/0 and blue 3/1/0. Saved reports:
`quality-head-indoor.json`, `quality-head-original.json`,
`quality-head-seed43-indoor.json`, `quality-head-seed43-original.json`.

## Reproduction

Prepare each queue with `PYTHONPATH=.:src .venv/bin/python
scripts/prepare_balloon_quality.py`, original bootstrap `--manifest`, the existing
indoor/expansion `--queue` and `--review`,
`--archive data/raw/engdes2-red-blue-v1-20261008/dataset.zip`, and a fresh `--output`.
Inspect every crop/full-frame sheet before approving a hash-bound review. Do not
reuse these decisions with a modified queue.

Fit the seed-42 head:

```sh
PYTHONPATH=.:src .venv/bin/python scripts/train_balloon_quality.py \
  --config configs/balloon-quality-training-20261008.json \
  --student-run runs/balloon-red-blue-new-views-fixed-teacher-20261008 \
  --output runs/balloon-quality-head-20261008
```

Repeat for the seed-43 parent and output. Existing outputs are refused.
Evaluate with `scripts/evaluate_balloon_quality.py --model .../quality.fp32.tflite
--source-report research/balloon-red-blue-20261008/new-views-fixed-teacher-indoor-scenes.json
--source-model new_views_fp32 --panel indoor --output <fresh report>`. Repeat for
the seed-43 report and corresponding `-original-scenes.json` with `--panel original`.

## Next work and boundaries

The new dataset and model path establish a tested way to supervise localization
separately from color/class confidence. The frozen linear head does not solve
the problem. A next bounded experiment is joint feature refinement with an
auxiliary localization objective and a class-preservation/distillation objective,
using training-only quality targets and a matched control. Do not choose new
quality thresholds against IMG frames or silently mutate classification behavior.
Compare both colors, both scene panels and both starting checkpoints before any
promotion. There are also two blue targets with no adequate proposal; a quality
head only ranks existing boxes and cannot repair missing proposals.

Representative independent Pi-camera recordings and live timing remain necessary.
The source groups and evaluation panels are not independent flight qualification.
The goal remains active and incomplete. No Pi, camera, SSH, network, ESP32 or motor
commands were issued. Existing concurrent changes were left untouched.

Verification: 30 new tests; full suite 459 passed, two skipped, six known TensorFlow
warnings. Scoped Ruff passed. Real exports passed parity checks and the original
runtime's contract rejection was verified. Implementation checkpoint: `8213176d`.
