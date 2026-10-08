# Combined training and broader scene diagnostic — 2026-10-08

## Decision

Do not promote the combined hard-negative candidate. It passes the tiny original
crop validation but detects **zero of eleven red balloons** in the additional
indoor scene panel. The fixed final epoch repeats that failure; selection of the
first epoch is not the sole explanation. No Pi weights or runtime defaults changed.

## Matched experiment

Both arms start from the same original 7,763-parameter separable student and use
78 previously reviewed external training crops repeated twice. Both have 412
training entries: 242 background, 92 red, 78 blue. The intervention repeats 11
reviewed training-only MSER negatives four times. The control substitutes 44
existing background entries. Seed 42, batch 16, eight epochs, learning rate 1e-4,
alpha 1 (hard labels, not new teacher-logit distillation), unchanged architecture.

Manifests, local and ignored:

- `data/balloon-red-blue-combined-control-20261008/manifest.json`:
  `326de18fc57e07bac9165a35c769b9a664bf4a6252734e510e3b340afcee7e0d`
- `data/balloon-red-blue-combined-refined-20261008/manifest.json`:
  `48c1ad3e5fd797e9ae45b2fcf2c96e0a16320dfabefb79af4844f11f9b47f745`

Runs use the same names under `runs/`. `student.keras` is selected by the original
eight-crop validation loss: control epoch 7, intervention epoch 1 (one-based).
`latest.keras` is the fixed epoch-8 endpoint, evaluated separately after the
regression appeared. Original checkpoints and exports remain intact. FP32 models
remain 35,204 bytes; INT8 models remain 15,656 bytes. Equal architecture does not
prove equal end-to-end latency when accepted proposals/tracking workloads change.

## Reviewed scene panel and limitations

The EngDes2 `IMG_` family is reserved from training across splits/augmentations.
Sixteen unique IMG identifiers were sampled from upstream validation before
predictions; twelve admitted after AI full-frame review. Excluded IDs 7, 9, 10,
15 have missing or ambiguous clipped/deflated targets. Decisions and reasons were
frozen before predictions in `configs/engdes2-development-scenes-20261008.json`.
Admitted truth contains 11 red and 12 blue balloons plus three negative frames.

This is **seen-environment development evidence**, not independent held-out
qualification. Rooms/props visibly resemble training frames despite different
filename families and passing the flip-aware perceptual duplicate screen.
The labels are AI-reviewed, not human-certified. Exclusions prevent a claim
covering every selected photo. Public export stretching/augmentation and generic
indoor photography are not representative Pi-camera competition conditions.
The original reserved test was not evaluated or admitted into training.

## Full-frame results

Threshold 0.8, one-to-one same-class IoU >=0.5, 320x240 processing, at most twelve
classification candidates. MSER with confirmed-parent part suppression remains
an experimental search variant, not an activated runtime default.

On the twelve new scenes, **FP32 with MSER confirmed-part suppression**:

| Model | Red TP/FP/FN | Red precision / recall | Blue TP/FP/FN | Blue precision / recall |
|---|---|---|---|---|
| Original student | 4/1/7 | 80.0% / 36.4% | 5/0/7 | 100% / 41.7% |
| External positives + background control | 7/1/4 | 87.5% / 63.6% | 7/3/5 | 70.0% / 58.3% |
| External positives + hard negatives | 0/0/11 | undefined / 0% | 5/0/7 | 100% / 41.7% |

Both trained INT8 exports produce the same counts as their FP32 counterparts on
this panel. This does not establish identical outputs everywhere. Epoch-8 exports
retain the same MSER counts as their selected checkpoints here. Baseline-search
counts and all detections are retained in the JSON receipts, including failures.

On the original four repeatedly used development photos, selected FP32 models
with the same MSER variant give control red 2/2/0, blue 3/1/0, and intervention red
2/2/0, blue 2/0/1. Intervention epoch 8 reduces red false positives to one, but
still misses one blue. Neither panel supports promotion or the 94% gate.

## Training fit and failure direction

On the 78 additional **training** crops, the control accepts 23/24 correct reds
and 22/24 correct blues; intervention accepts 17/24 reds and 9/24 blues. Final
epoch intervention accepts 17/24 and 8/24. All four exports accept zero of the
30 additional background crops. This is fit, not generalization.

On the original 11 hard negatives, FP32 accepted detections decrease from 11
(control) to seven (selected intervention) and three (final intervention), while
new positive recall collapses. This demonstrates a harmful operating tradeoff;
the added negatives are not a proven benefit merely because they reduce false
positives on their own training crops.

Inspection of the control's new-panel MSER candidates finds an IoU>=0.5 proposal
for every red target, but only 10/12 blues. Seven targets of each color are
accepted. Therefore both classifier rejection and search coverage need work;
neither compression nor a global threshold change alone addresses this result.

Next: review/train on proposal-shaped crops from the admitted **training-only**
indoor frames, investigate the positive/negative representation conflict, and
reassess on both development panels. Do not mine the new panel into training or
use it for a final 94% claim. Representative independent real recordings and live
Pi sustained-latency qualification remain outstanding. No ESP32/motor activity.

## Reproducible evidence

- `combined-indoor-scenes.json`: selected FP32/INT8 and original FP32, new panel.
- `combined-original-scenes.json`: both selected exports, old development photos.
- `combined-final-indoor-scenes.json`: fixed final epoch FP32, new panel.
- `combined-final-original-scenes.json`: fixed final epoch FP32, old photos.
- `combined-training-fit.json`: all four FP32 checkpoints on external train crops.

Receipts bind model, review and source hashes. New evaluator checks exact review
coverage, explicit evaluation-only approval, frame hashes, path containment,
coordinate validity and the frozen threshold. The dataset builder rejects the
reserved IMG family and hard negatives outside original training source/session
pairs or conflicting with positive/held-out crop hashes.

Verification: full suite **350 passed, 2 skipped**, six existing TensorFlow
warnings; scoped Ruff checks passed. Engineering tests do not qualify perception.
