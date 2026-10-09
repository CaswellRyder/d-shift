# Visible-balloon box refinement — 2026-10-08

## Decision

Research only; do not promote. Two frozen-feature box regressors fit training
boxes better but worsen average box IoU on the head-label holdout. Seed 43
recovers one red and one blue indoor detection with corrected-box suppression,
but adds two red false positives on the original-photo panel. Seed 42 counts
are unchanged. A post-hoc frozen-selection control removes both the gains and
the extra false positives. The experiment identifies instance/part selection
as a continuing problem; it does not justify deployment or flight readiness.

No Pi, network, camera, goal-model, ESP32 or actuator changes were made.

## Method and boundaries

Reuse each existing new-views seed-42/43 student's 128 `spatial_features`.
Freeze the entire original classifier. Attach a four-output affine head to
predict the visible target's crop-normalized x1/y1/x2/y2 coordinates. Four fixed
ridge fits (penalty 10) fold feature standardization into the head weights.
No shared-feature training, class relabeling, threshold adjustment or second
image inference is required. This extends the same tiny separable-context
student distilled from the desktop MobileNetV4 teacher, not a new full V4
network deployed on the Pi.

Source: the existing hash-bound, explicitly reviewed 144-proposal quality
collection. Reuse its strict training-source, crop/frame hash, annotation,
review and IMG-family guards. The matched annotation must overlap the original
proposal by at least 0.25 IoU, beat any other known target by at least 0.10 IoU,
and lie entirely inside the visible padded input crop. This avoids fitting a
box around an invisible continuation or an ambiguous target.

- 84 examples fit the head; 18 examples in deterministic source-group fold 0
  are held out from box-head fitting.
- Excluded: 16 reviewed negatives, 17 low-overlap proposals, seven targets
  outside the crop, two ambiguous matches.
- Held-out source groups, source names and pixel hashes cannot cross into the
  head's training set. The original backbone has seen related classification
  imagery: this is **not** independent full-model qualification.
- All IMG-family images remain excluded from fitting, calibration and negative
  mining. Development IMG scenes are used for evaluation only. Original reserved
  test images remain unevaluated. Labels are agent-reviewed, not human-certified.
- No new synthetic images or data downloads were used in this experiment.

At inference, normalized coordinates are clipped to [0,1]. A decoded box must
have at least four pixels on each side, area between 0.25 and 4 times the original
box, and at least 0.10 IoU with it. Otherwise use the original box. Only raw-
accepted balloon candidates receive a correction; the crop and class score
remain unchanged. These fixed bounds were set before development evaluation,
not tuned to make the reported scene counts pass.

Code: `scripts/train_balloon_box_refinement.py` and
`scripts/evaluate_balloon_box_refinement.py`.
Config: `configs/balloon-box-refinement-20261008.json`, SHA-256
`89360ac7c6ee1a4a3fb3516dca975bdea3fe2bbcd7609593b52bee848ba6af01`.

## Training and held-out box evidence

The original train proposals have mean IoU 0.81384 and localize 79/84 reviewed
targets at IoU >=0.5. Their 18 eligible held-out proposals already localize
18/18, with mean IoU 0.87128. This selected subset mostly has good boxes and
does not measure missing-proposal recall.

| Seed | Refined train mean IoU | Train >=0.5 | Refined holdout mean IoU | Holdout >=0.5 |
| --- | ---: | ---: | ---: | ---: |
| 42 | 0.87433 | 84/84 | 0.78593 | 17/18 |
| 43 | 0.85575 | 84/84 | 0.82214 | 18/18 |

Both heads improve five held-out boxes and worsen thirteen. None of these 102
train/holdout examples trigger the bounded fallback. The gap between fitted and
held-out localization is evidence against claiming generalization from this
small linear head, even where end-to-end scene counts improve.

## Full-scene comparison

Use the same 59 indoor proposals across 12 frames and 31 original-panel
proposals across four photos as each parent's previous report. Fresh FP32 class
probabilities are **exactly identical** to parent values in all four comparisons.
Fresh duplicate/confirmed-part suppression runs on either original or corrected
boxes. The uncorrected arm reproduces the original parent metrics exactly.
Acceptance remains 0.8; ground-truth matching remains same-class IoU >=0.5.

Counts are TP / FP / FN:

| Seed | Panel | Arm | Red | Blue |
| --- | --- | --- | --- | --- |
| 42 | Indoor | Parent and refined | 11 / 0 / 0 | 9 / 4 / 3 |
| 42 | Original | Parent and refined | 2 / 2 / 0 | 3 / 0 / 0 |
| 43 | Indoor | Parent | 10 / 2 / 1 | 8 / 4 / 4 |
| 43 | Indoor | Refined | 11 / 2 / 0 | 9 / 4 / 3 |
| 43 | Original | Parent | 2 / 2 / 0 | 3 / 1 / 0 |
| 43 | Original | Refined | 2 / 4 / 0 | 3 / 1 / 0 |

The indoor gains occur in the already-reviewed IMG_5180 and IMG_5432 development
frames. Do not mine these into training. The extra red false positives are in
original validation photo `balloon/train/2685563244_b0d5f7eb67_b.jpg` (the archive
path says train; our manifest assigns it to validation). Two previously
suppressed same-color parts survive after corrected geometry changes the
containment decision. Their original boxes are `[100,203,133,224]` and
`[127,200,150,219]`.

### Post-hoc fixed-selection diagnostic

After seeing that regression, `scripts/compare_box_selection.py` replays the
same corrected boxes but preserves every original winner/rejection. No extra
neural inference, training or threshold tuning occurs. The script verifies
matching proposal identities, class scores, frame truth and parent counts.

**All four comparisons return exactly to their original TP/FP/FN counts.** This
removes the two extra false positives but also both indoor gains. Thus the
reported detection gains require a different selected set of proposals; merely
correcting the original selected boxes does not cross the matching threshold
for another target on these panels. This policy was investigated after inspecting
development results and is explicitly exploratory, not an independent test.

Per-candidate reports: `box-refinement*-scenes.json`; fixed-selection reports:
`box-refinement*-scenes-locked-selection.json`. Each report binds its source
report/model and code hashes. A filename family or source-group split is not
proof of independent recording sessions or unseen environments.

## Footprint, provenance and checks

The head adds 516 parameters, for **8,279 total**. Each FP32 TFLite model is
**37,520 bytes** versus the original parent's 35,204 bytes (+2,316 bytes).
There is one backbone invocation, but actual ARMv6 latency, capture freshness
and energy remain unmeasured. No INT8 calibration/export or Pi speed claim.

- Seed 42: `runs/balloon-box-refinement-20261008/box.fp32.tflite`, SHA-256
  `be115c114aabfd3b015af08720a3965fef5c44d5cdee651bf9aafc13078c41ad`.
- Seed 43: `runs/balloon-box-refinement-seed43-20261008/box.fp32.tflite`, SHA-256
  `feccd90a5e6f0cf5de542a80d04c109a64e3be77c14ad9481e8b99395dde23ec`.
- Frozen Keras logits match the parent exactly. Maximum raw-output export
  discrepancy on all 102 fitted/held-out crops is 8.58e-6 / 4.77e-6.
- Seven-output research metadata and a dedicated predictor prevent treating
  box coordinates as extra class logits. The regular runtime is not modified.
- Run folders retain Keras checkpoints, metadata and per-example training/holdout
  reports; model artifacts and data remain ignored by version control.
- 32 new focused tests cover target exclusions, source separation, decode bounds,
  class parity, model contracts and locked-selection identity checks. Full suite:
  **517 passed, 2 skipped**, six existing TensorFlow warnings. Scoped Ruff passes.

Reproduce with new output paths (existing evidence is never overwritten):

```sh
PYTHONPATH=.:src .venv/bin/python scripts/train_balloon_box_refinement.py \
  --config configs/balloon-box-refinement-20261008.json \
  --student-run runs/balloon-red-blue-new-views-fixed-teacher-20261008 \
  --output runs/balloon-box-refinement-reproduction

PYTHONPATH=.:src .venv/bin/python scripts/evaluate_balloon_box_refinement.py \
  --model runs/balloon-box-refinement-reproduction/box.fp32.tflite \
  --source-report research/balloon-red-blue-20261008/new-views-fixed-teacher-indoor-scenes.json \
  --source-model new_views_fp32 --panel indoor \
  --output runs/balloon-box-refinement-reproduction/indoor.json

PYTHONPATH=.:src .venv/bin/python scripts/compare_box_selection.py \
  --report runs/balloon-box-refinement-reproduction/indoor.json \
  --output runs/balloon-box-refinement-reproduction/locked.json
```

For seed 43 select its matching parent and scene report; the original panel
uses `--panel original` and its matching report. The fixed research config is
deliberately not a parameter-sweep interface.

Next work should distinguish a balloon part from a nearby same-color instance
using permissible training-only sources and compare against the frozen parent.
The present results do not support deploying a box-correction head merely
because one repeatedly-inspected development subset improved.
