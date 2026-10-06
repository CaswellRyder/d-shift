# Reviewed-label goal detector refinement — 2026-10-05

Status: completed at the unchanged 12-epoch cap; all training/evaluation processes terminal.
Epoch 2 is strongest by the unchanged weakest-metric ranking, but orange-triangle precision
and yellow-square recall still fail. Epoch 1's yellow pass did not persist to epoch 2.
This is local offboard YOLO11n research, not MobileNet distillation, a Pi deployment,
or flight authorization.

Startup verification: trainer reports all 499 state entries transferred. Comparing the
captured pre-forward model against the parent EMA cast to FP32 gives exact equality for
all 499 entries; initialization did not silently revert to COCO weights. Runtime profile
verification passed. No accuracy improvement is inferred from these checks.

## Completed refinement: no passing checkpoint

`comparison-completed-refinement.json` compares all 12 refinement epochs, finalized native
best, and baseline epoch 17. Refinement epoch 2 remains first by the unchanged weakest-class
metric ranking. No checkpoint meets all six per-class precision/recall requirements.

Final epoch 12 (`interim-l/development-640.json`):

| Class | Precision | Recall |
| --- | ---: | ---: |
| Orange circle | 95.47% | 92.80% |
| Orange square | 94.29% | 97.06% |
| Orange triangle | 83.84% | 95.52% |
| Yellow circle | 84.53% | 96.23% |
| Yellow square | 96.61% | 89.41% |
| Yellow triangle | 98.33% | 99.16% |

**FAIL:** orange-triangle precision, yellow-circle precision and yellow-square recall.
Final snapshot SHA256: `1b26d420c20f429f3b88cd9ed360885b9074087033b130d8d4124438f23eedc8`.
The finalized native-best report (`completed-best-development-640.json`) reproduces epoch
10's fixed-operating-point counts and also fails. Native best SHA256:
`60e27241bff3977c21afc8e0816a586cb7f0408c15a2f34e9983105a8cdc5454`.
Native best mAP50-95 is 0.73081 at epoch 10; it does not supersede the per-class standard.

Next work follows the [error/coverage review](GOAL_ERROR_FOLLOWUP.md), not another extension
of this run. Close-range training coverage and partial-triangle annotation consistency need
attention. Validation labels and thresholds remain fixed; the reserved test is untouched.
No distillation, viewer replacement, Pi deployment or flight qualification is granted.

## Development results: epochs 10 and 11

Both saved boundary snapshots were evaluated on all 595 unchanged development images at
640 pixels on CPU, confidence 0.25 and matching IoU 0.5.

| Class | Epoch 10 precision / recall | Epoch 11 precision / recall |
| --- | ---: | ---: |
| Orange circle | 95.45% / 92.40% | 95.47% / 92.80% |
| Orange square | 93.84% / 97.06% | 94.29% / 97.06% |
| Orange triangle | 83.84% / 95.52% | 83.48% / 95.52% |
| Yellow circle | 90.00% / 96.23% | 90.00% / 96.23% |
| Yellow square | 91.60% / 89.80% | 93.06% / 89.41% |
| Yellow triangle | 98.33% / 99.16% | 98.33% / 99.16% |

Both **FAIL** orange-triangle precision and yellow-square recall. Results are retained in
`interim-j/development-640.json` and `interim-k/development-640.json`.
`comparison-through-epoch11.json` still ranks refinement epoch 2 first, baseline epoch 17
second. Nothing is promoted. The unchanged 12-epoch schedule is not extended.

A [cached-error and coverage review](GOAL_ERROR_FOLLOWUP.md) identifies concentrated
close-range yellow-square misses and orange-triangle localization/annotation concerns.
These findings do not change labels or measured scores. Verification rerun: 135 tests pass,
two NMS tests skip in the main environment and pass separately in the detector environment;
four existing TFLite warnings remain. Ruff passes.

## Development result: epoch 9

`interim-i/development-640.json`, unchanged full CPU validation:

| Class | Precision | Recall |
| --- | ---: | ---: |
| Orange circle | 95.51% | 93.60% |
| Orange square | 93.40% | 97.06% |
| Orange triangle | 83.84% | 95.52% |
| Yellow circle | 85.96% | 96.23% |
| Yellow square | 92.31% | 89.41% |
| Yellow triangle | 98.33% | 99.16% |

**FAIL:** orange-triangle precision, yellow-circle precision and yellow-square recall.
`comparison-through-epoch9.json` retains epoch 2 first. Native best fitness remains at epoch 8.
Checkpoint SHA256: `c378935c6c52fe10e10b070321459aab71a46d91fe12a325e53ed1aca5f451f9`.
Epoch 10 startup verified full-precision training-state/profile restoration and RNG sidecar
`68b9ff2c34fbf80bf1f68dec30c6e777e40dcaf0ac9d15889acf4c4a0918439a`.
The separate 960-input experiment below also fails and is not promoted.

## Development result: epoch 8

`interim-h/development-640.json`, same fixed-confidence CPU evaluation of all 595 frames:

| Class | Precision | Recall |
| --- | ---: | ---: |
| Orange circle | 95.49% | 93.20% |
| Orange square | 93.84% | 97.06% |
| Orange triangle | 81.36% | 95.52% |
| Yellow circle | 90.00% | 96.23% |
| Yellow square | 91.90% | 89.02% |
| Yellow triangle | 98.33% | 99.16% |

**FAIL:** orange-triangle precision and yellow-square recall. Yellow-circle precision is
exactly 153/170 = 90%, not a rounded-up pass. `comparison-through-epoch8.json` retains epoch 2
first; native best fitness 0.72561 at epoch 8 remains a different selection criterion.
Checkpoint SHA256: `79f47a61d61246fedcd7134cafa136c4e9df4717a727fcfcdcff3e82c8871283`.
Epoch 9 startup verified exact training-state/profile restoration and restored RNG sidecar
`9a7b153801f21246498d77b025474d9ecb7520e5478a61eabdb832cbab4c61af`.
No threshold changes, test inference or promotion. The experiment still ends by epoch 12.

## Development result: epoch 7

`interim-g/development-640.json`, unchanged full CPU validation:

| Class | Precision | Recall |
| --- | ---: | ---: |
| Orange circle | 95.45% | 92.40% |
| Orange square | 93.84% | 97.06% |
| Orange triangle | 82.40% | 95.52% |
| Yellow circle | 86.93% | 96.23% |
| Yellow square | 95.83% | 90.20% |
| Yellow triangle | 98.33% | 99.16% |

**FAIL:** orange-triangle and yellow-circle precision. `comparison-through-epoch7.json`
retains epoch 2 first. Native best fitness improves to 0.72461 at epoch 7, but that is not
the fixed-operating-point class acceptance criterion. The original 12-epoch cap remains.
Checkpoint SHA256: `b3912d7737e6cb3aa938e20b8e76ec970eb1d87ed7f77153fea748b96aca56af`.
Epoch 8 startup verified full-precision state/profile restoration and restored RNG sidecar
`a4ce56d2262f1e2dc188dc85cd80f4a36f6f2ecb3bfe9f9d58986e4a2b4ee1fd`.

## Development result: epoch 6

`interim-f/development-640.json`, unchanged full CPU validation:

| Class | Precision | Recall |
| --- | ---: | ---: |
| Orange circle | 95.42% | 91.60% |
| Orange square | 93.84% | 97.06% |
| Orange triangle | 80.33% | 95.52% |
| Yellow circle | 82.70% | 96.23% |
| Yellow square | 90.49% | 93.33% |
| Yellow triangle | 98.33% | 99.16% |

**FAIL:** orange-triangle and yellow-circle precision. Higher yellow-square recall does not
offset the orange regression. `comparison-through-epoch6.json` retains epoch 2 first; no
passing checkpoint, label changes or viewer promotion. Native best fitness remains at epoch 2.
Checkpoint SHA256: `fe60c48bf38f2755716c27ba83e7aa0d0548fa33cf1d193523c0a2ba51452a40`.
Saved full-state/RNG sidecar SHA256:
`682772ac1e4e0b51145a560063f98297e9600ae1c63c6ad82e9d57870e32b7f2`.

## Development result: epoch 5

`interim-e/development-640.json`, unchanged full CPU validation:

| Class | Precision | Recall |
| --- | ---: | ---: |
| Orange circle | 95.04% | 92.00% |
| Orange square | 94.29% | 97.06% |
| Orange triangle | 82.40% | 95.52% |
| Yellow circle | 81.82% | 96.23% |
| Yellow square | 97.81% | 87.45% |
| Yellow triangle | 98.33% | 99.16% |

**FAIL:** orange-triangle precision, yellow-circle precision, yellow-square recall.
`comparison-through-epoch5.json` retains epoch 2 first. Later training has not improved the
weakest required metric; keep the existing patience-eight rule, not an extended schedule.
Checkpoint SHA256: `065cc62570975f3aec2e3730a3395336139ba4ae34fbe7204b9d1422feba6265`.
Epoch 6 startup verified exact full-precision restoration, the same refinement options,
and restored RNG sidecar `f66b5988cb8a78830b63db354d709199edd0beff58d6fb6662e10551306de057`.

## Development result: epoch 4

`interim-d/development-640.json`, same full CPU validation and fixed operating point:

| Class | Precision | Recall |
| --- | ---: | ---: |
| Orange circle | 95.44% | 92.00% |
| Orange square | 92.52% | 97.06% |
| Orange triangle | 83.91% | 96.02% |
| Yellow circle | 83.15% | 96.23% |
| Yellow square | 91.02% | 87.45% |
| Yellow triangle | 98.33% | 99.16% |

**FAIL:** orange-triangle precision, yellow-circle precision, yellow-square recall.
`comparison-through-epoch4.json` retains epoch 2 as strongest. Native best fitness remains
0.72212 at epoch 2; the existing patience-eight stopping rule is preserved. No promotion.
Checkpoint SHA256: `b42bffb108ef429e36fad0746019cc413ff8c9c6fbf500bae4fb00d477b1de89`.
Epoch 5 startup verifies the same refinement profile, exact full-precision training-state
restoration, and RNG sidecar SHA256
`3b6f04175a15ca19b0af44fd4f9ce5251127cab5eb828aac093f99943e7e95be`.

## Development result: epoch 3

`interim-c/development-640.json`, unchanged 595-frame CPU evaluation:

| Class | Precision | Recall |
| --- | ---: | ---: |
| Orange circle | 93.47% | 91.60% |
| Orange square | 92.96% | 97.06% |
| Orange triangle | 85.65% | 95.02% |
| Yellow circle | 84.07% | 96.23% |
| Yellow square | 91.27% | 90.20% |
| Yellow triangle | 98.33% | 99.16% |

**FAIL:** orange-triangle and yellow-circle precision. Yellow-square recall recovered, but
the new yellow-circle false positives make this checkpoint weaker than epoch 2. The unchanged
ranking in `comparison-through-epoch3.json` retains epoch 2 first. Both color aggregates
exceed 90%; that does not override individual-class failures. Epoch 4 startup verified exact
training-state and RNG restoration (sidecar SHA256
`1fee64f8c55420d99726c5b26c34d4ae97cb01ade49c926d6416f98fc8303e7b`).

## Strongest development checkpoint: epoch 2

`interim-b/development-640.json`, same 595 frames, CPU 640, confidence 0.25 / IoU 0.5:

| Class | Precision | Recall |
| --- | ---: | ---: |
| Orange circle | 94.24% | 91.60% |
| Orange square | 92.92% | 96.57% |
| Orange triangle | 86.88% | 95.52% |
| Yellow circle | 96.84% | 96.23% |
| Yellow square | 96.20% | 89.41% |
| Yellow triangle | 98.33% | 99.16% |

**FAIL:** orange-triangle precision and yellow-square recall. `comparison-through-epoch2.json`
ranks this whole checkpoint ahead of baseline epoch 17 and refinement epoch 1 by the
pre-existing weakest-required-metric rule. This is not the same as counting passing classes:
epoch 1 passed five classes, while epoch 2 passes four but has a higher weakest metric.
No individual class scores are combined across checkpoints.

Checkpoint SHA256: `69a694e2e5acfb666fe679288173499fd8ed44eed7c50c0017bcb00badc79bce`.
Epoch 3 startup verified exact full-precision state restoration and the unchanged refinement
profile; RNG sidecar SHA256 `4030dc2b50985bae92f32423966273e064d6ddd09d6fc442cf683127260e5feb`.
Free disk was about 21 GiB. The reserved test, validation annotations and viewer are unchanged.

## First development result

`runs/goal-detector-refined-20261005/interim-a/development-640.json` evaluates all 595
development frames on CPU at confidence 0.25 / IoU 0.5. No threshold change.

| Class | Precision | Recall |
| --- | ---: | ---: |
| Orange circle | 93.78% | 90.40% |
| Orange square | 92.45% | 96.08% |
| Orange triangle | 86.04% | 95.02% |
| Yellow circle | 91.07% | 96.23% |
| Yellow square | 97.08% | 91.37% |
| Yellow triangle | 98.33% | 99.16% |

**Overall FAIL: orange-triangle precision.** All three yellow classes pass this development
check; yellow-square recall rises from the parent's 88.63% to 91.37%. This is not a uniform
improvement: orange-circle recall and yellow-circle precision decline, and the weakest-metric
ranking still prefers baseline epoch 17 (86.55% weakest metric vs 86.04% here).
Do not combine their individual class results into a fictional passing model.

Cached error diagnostics partition orange-triangle's 31 false positives into 18 same-class
localization overlaps below IoU 0.5 and 13 with no labeled overlap. These are descriptive
categories, not proof of label errors, and no validation annotations were changed.
Model SHA256: `07777fad18aa4684864f73a7331423d4e196f3cfa1e161dcf0711e212bd5aa32`.

## Rejected inference-size experiment: epoch 2 at 960

`interim-b/development-960.json` uses the exact epoch-2 checkpoint, all 595 original
validation frames and unchanged confidence/IoU/NMS settings, but requests 960-square detector
input instead of 640. This upsamples the existing source pixels; it adds no image detail.
An independent single-frame check confirmed predictor input size [960, 960] with source
shape (640, 640), rather than merely recording a requested size that was ignored.
The evaluator now permits 960 explicitly, with 640 still the default. This is an offboard
experiment, not Pi latency or compatibility evidence.

| Class | Precision at 960 | Recall at 960 |
| --- | ---: | ---: |
| Orange circle | 96.22% | 91.60% |
| Orange square | 85.09% | 95.10% |
| Orange triangle | 87.16% | 94.53% |
| Yellow circle | 98.08% | 96.23% |
| Yellow square | 96.58% | 88.63% |
| Yellow triangle | 98.33% | 99.16% |

**Rejected:** orange-triangle precision remains below 90%; orange-square precision now fails,
and yellow-square recall also fails. The small triangle-precision gain does not justify the
new failure. No configuration is promoted. Raw-checkpoint ranking remains restricted to
matching input sizes; this experiment is deliberately not mixed into the 640 ranking.
Six comparison tests and Ruff pass after the CLI extension.

### Validation-label caveat (diagnostic review, not relabeling)

The generated `interim-a/review/orange_triangle-unmatched.jpg` shows the highest-confidence
unmatched detections, not an unbiased sample. Two full original validation images were then
visually inspected:

- `20241002_192159_012_jpg.rf.faee53a2ec9dd4d222fce03bb91db4bc.jpg`: an orange triangular
  goal is visible beneath the far railing near x=282, y=207. The supplied frame has five goal
  annotations but no orange-triangle annotation. The detector's box is approximately
  [273.91, 197.89, 291.31, 218.23], confidence 0.8122; it is counted as a false positive.
  Image SHA256: `72cd6db08f65cf01895d0ab5b2830a1d8aa1225afbed8acf45e0501ef001a374`.
- `20241002_192203_005_jpg.rf.d91df927a5971ea6e886a106fbcab78c.jpg`: an orange triangular
  goal is partially clipped by the top image edge near x=118. There are no supplied goal
  annotations in this frame. Whether this clipped object should be labeled requires an
  explicit annotation policy; its detection is not automatically accepted as correct.
  Image SHA256: `50575c2f47e45b2876bf8d7a9dfac71e27c8e6f80c4550d9578560bce80a6851`.

This supports a validation-label completeness concern, not an adjusted accuracy claim.
Some other unmatched boxes sit just below IoU 0.5 (e.g. 0.4791); that alone does not prove
the supplied box is wrong. No labels, thresholds, or counts were changed. Any audited
evaluation set must be versioned and reviewed beyond model-selected errors, and cannot
replace independent qualification with a post-hoc pass.

## Fixed experiment

- Output: `runs/goal-detector-refined-20261005` (separate from completed baseline).
- Initialization: `runs/goal-detector-safe-20261005/interim-q`, baseline epoch 17.
  Model SHA256: `81cae2f93e5f3b8ff6a1de53e9124002e0643046fdf870aaf016594bda80c124`.
- Data: `data/goal-detector-v10-reviewed-color-20261005`, 2,950 training frames,
  2,826 goal boxes; 595 byte-identical development-validation frames.
  Receipt SHA256: `f88cff3e4a0e1d1b75acf988bca62a57c35d7d8a519f1bdea056a062702ef282`.
- Label review: five duplicate/conflicting training annotations removed; one yellow goal
  relabeled from orange, one false goal annotation on a green balloon removed. Original
  data and all validation annotations preserved. Review files record visual decisions.
- New optimizer: AdamW, LR 0.0001, final LR fraction 0.01, warmup 0, mosaic 0.
  Other baseline settings retained, including 640 input, batch 8, seed 42, patience 8.
- Maximum 12 epochs. Saved one-epoch process boundaries preserve full-precision
  model/EMA/optimizer/scheduler/scaler and random streams. Exit 75 means resume required.
  Same 10-GiB disk guard. Restarts are not bitwise uninterrupted-training equivalence.
- Acceptance unchanged: each of six classes needs precision AND recall >=90% at confidence
  0.25 and IoU 0.5. Parent fails orange-triangle precision and yellow-square recall.

This experiment changes initialization, labels and optimization/augmentation together.
It cannot establish which individual change caused a difference. Selecting the parent on
development validation also means future development gains are not independent qualification.
Reserved test and real neon-goal camera/Pi measurements remain separate gates.

## Commands

Run from the project directory. Initial command has already been launched; do not rerun it:

```sh
PYTHONPATH=src .detector-venv/bin/python scripts/train_goal_detector.py \
  --data data/goal-detector-v10-reviewed-color-20261005 \
  --output runs/goal-detector-refined-20261005 --epochs 12 --batch 8 \
  --refine-from runs/goal-detector-safe-20261005/interim-q --max-process-epochs 1
```

Only after the current process exits 75 and its checkpoint is saved:

```sh
PYTHONPATH=src .detector-venv/bin/python scripts/train_goal_detector.py \
  --data data/goal-detector-v10-reviewed-color-20261005 \
  --output runs/goal-detector-refined-20261005 --epochs 12 --batch 8 \
  --resume --max-process-epochs 1
```

Resume recovers the recorded refinement profile; do not pass `--refine-from` again.
Never overlap writers. Do not resume a completed run. Evaluate immutable snapshots using
the same CPU 640 fixed-confidence evaluator and compare against baseline epoch 17.
No viewer promotion or automatic distillation follows from training completion.
