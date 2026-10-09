# Shared-feature localization refinement — 2026-10-08

## Decision

Do not promote these models. Joint refinement did not improve full-scene
detection over its matched frozen-feature control. Seed 43 added one blue false
positive; seed 42 had unchanged counts. Quality-ranked containment still did
not outperform confidence-ranked containment. The Pi deployment, runtime
contract, goal models, camera, ESP32 and motor interface were not changed.

This extends [the frozen quality-head experiment](QUALITY_HEAD.md). It does not
establish 94% precision/recall or flight readiness. Both development panels have
been examined repeatedly; neither is a final qualification set.

## Controlled experiment

- Parents: the existing 64-pixel separable-context students distilled from the
  fixed MobileNetV4 teacher, seeds 42 and 43. Start from each original
  `student.keras`, not the previously fitted quality model.
- Add a 129-parameter affine quality head to the existing 128 shared features.
  Outputs are three class logits plus a raw box-IoU estimate, clipped to [0,1]
  for ranking. This is a research-only four-output model, not a drop-in runtime
  classifier.
- Joint arm: train shared features and quality head, freeze classifier weights.
  Control arm: train only the quality head, freeze the entire parent network.
- Initialize each quality head with ridge penalty 10 using only this run's
  quality training partition. Both arms use the same seed-specific initializer,
  data, shuffled minibatches and optimization schedule.
- Fixed ten final epochs; Adam 0.0001; batch 16; temperature 4. The objective is
  temperature-scaled KL preservation of the parent class distribution plus
  quality MSE on labeled quality crops. Rescale the sparse quality loss by
  total examples / quality examples. No new class-label cross-entropy loss.
- No development-panel or quality-holdout checkpoint selection. No threshold
  tuning: class acceptance stays at 0.8; scene matching stays at IoU 0.5.

Configuration: `configs/balloon-quality-joint-20261008.json`.
Implementation: `scripts/refine_balloon_quality.py`.

## Data boundaries

The reviewed quality collection contains 144 candidate crops from 38 source
groups. Deterministic SHA-256 group fold 0 holds out 33 crops from eight groups;
111 quality crops remain for refinement. Every class-training entry sharing a
held-out source name, source group or source-pixel hash is also excluded,
leaving 508 class-preservation entries and 619 total refinement entries.

The eight held-out groups are `frame_000003_jpg.jpg`, `frame_000203_jpg.jpg`,
`frame_000387_jpg.jpg`, `frame_000448_jpg.jpg`, `frame_000530_jpg.jpg`,
`frame_000660_jpg.jpg`, `frame_0570_jpg.jpg`, and `frame_0639_jpg.jpg`.

These quality labels are withheld from **this refinement only**. The original
parent already saw related classification imagery; this is not an independent
end-to-end generalization test. The labels were visually reviewed by the agent,
not independently certified by a human. Unknown proposals remain unknown rather
than becoming background examples.

All `IMG_` family images remain excluded from training, calibration and negative
mining. The indoor `IMG_` panel is development evaluation only. The original
reserved test remains separate and unevaluated. No synthetic images were added
in this experiment.

## Training and export results

MAE is absolute error in the clipped IoU estimate, not detection accuracy.

| Seed | Arm | Initial train MAE | Final train MAE | Quality-label holdout MAE |
| --- | --- | ---: | ---: | ---: |
| 42 | Frozen control | 0.084491 | 0.084613 | 0.174696 |
| 42 | Joint | 0.084491 | 0.080371 | 0.179825 |
| 43 | Frozen control | 0.088409 | 0.087515 | 0.132311 |
| 43 | Joint | 0.088409 | 0.083706 | 0.131347 |

Joint training reduced fitting error, but held-out quality error worsened for
seed 42 and improved only slightly for seed 43. All four arms classified the
same eight original validation crops correctly (8/8), a tiny crop-only check.
Neither joint arm changed any training-crop argmax class. Both controls retained
all backbone weights exactly; all four retained classifier weights exactly.

Each model has 7,892 parameters and a 35,972-byte FP32 TFLite export. Maximum
Keras/TFLite raw-output discrepancy across all 619 training crops was below
0.000009 in every run. These exports were tested on the Mac, not on ARMv6; no new
Pi throughput, capture latency, power or freshness measurements are available.
Training ran with TensorFlow GPU visibility disabled. The converter additionally
printed Metal device initialization logs; those are not Pi evidence.

## Scene results

Replay uses the same hash-bound saved proposals as each parent report, fresh
class and quality inference, and **fresh ordinary duplicate suppression**.
Old suppression flags are not carried over when class scores change. Then three
containment policies are compared. The separate evaluator intentionally does
not weaken the original frozen-feature evaluator's class-parity checks.

Counts below are TP / FP / FN. Indoor panel: 12 reviewed 320x240 frames,
11 red and 12 blue balloons, 59 candidate evaluations per model.

| Seed | Arm | Policy | Red | Blue |
| --- | --- | --- | --- | --- |
| 42 | Control | Parent-first | 11 / 0 / 0 | 9 / 4 / 3 |
| 42 | Joint | Parent-first | 11 / 0 / 0 | 9 / 4 / 3 |
| 42 | Control | Confidence or quality | 11 / 0 / 0 | 8 / 5 / 4 |
| 42 | Joint | Confidence or quality | 11 / 0 / 0 | 8 / 5 / 4 |
| 43 | Control | Parent-first | 10 / 2 / 1 | 8 / 4 / 4 |
| 43 | Joint | Parent-first | 10 / 2 / 1 | 8 / 5 / 4 |
| 43 | Control | Confidence or quality | 11 / 1 / 0 | 8 / 4 / 4 |
| 43 | Joint | Confidence or quality | 11 / 1 / 0 | 8 / 5 / 4 |

The seed-43 regression is a proposal at `[61, 60, 248, 156]` in development
frame `IMG_7046_00217_jpg.rf.944dc7af9da501f0e4f4d6e51a522464.jpg`. Its blue score
increased from 0.780364 to 0.823069, crossing the fixed 0.8 threshold and adding
a false positive. This observation is diagnostic only: that frame must not
be mined into training.

Original panel: four validation photos, two red and three blue balloons,
31 candidate evaluations per model. All arms and all policies retain red
2 / 2 / 0; seed 42 retains blue 3 / 0 / 0 and seed 43 retains blue 3 / 1 / 0.

The controls reproduce parent scores exactly on both panels (maximum probability
delta 0). Joint maximum deltas on indoor/original panels are 0.069197/0.055807
(seed 42) and 0.053521/0.017546 (seed 43). No candidate argmax labels change;
only the seed-43 acceptance crossing above changes raw acceptance.

Full per-candidate evidence is retained in `joint-quality*-scenes.json` beside
this document. Each file binds its model, metadata, source report, evaluator
and helper code hashes. These are fixed-proposal development replays, not a
fresh live-camera benchmark or competition simulation.

## Artifact identities and reproduction

Ignored run folders contain provenance, progress, epoch CSV, report, Keras
checkpoint and FP32 TFLite model. Prefix is `runs/balloon-quality-joint` and
suffix is `-20261008`:

| Infix | Arm/seed | FP32 SHA-256 |
| --- | --- | --- |
| (none) | Joint 42 | b48cb27300cbe0cf36e5b1ec8e44604ec75b8a922841605a82b43ad1b36272b7 |
| `-control` | Control 42 | 586fa6391ecff33770dd3b96be13b54fcbca6c2c4a659971fb45825757e2997e |
| `-seed43` | Joint 43 | 691532503b8695d51a0c5f9ba74e7da8339fddeea1ce139f606f15da39d674ff |
| `-control-seed43` | Control 43 | adda37f586b8de442d7e438eeed3ca25b1b9a109f5c33ed7d26a532bb16e642f |

Example commands (choose a new output path; existing evidence is not overwritten):

```sh
PYTHONPATH=.:src .venv/bin/python scripts/refine_balloon_quality.py \
  --config configs/balloon-quality-joint-20261008.json \
  --student-run runs/balloon-red-blue-new-views-fixed-teacher-20261008 \
  --mode joint --output runs/balloon-quality-joint-reproduction

PYTHONPATH=.:src .venv/bin/python scripts/evaluate_joint_balloon_quality.py \
  --model runs/balloon-quality-joint-reproduction/quality.fp32.tflite \
  --source-report research/balloon-red-blue-20261008/new-views-fixed-teacher-indoor-scenes.json \
  --source-model new_views_fp32 --panel indoor \
  --output runs/balloon-quality-joint-reproduction/indoor-scenes.json
```

For the control use `--mode head_only` with a separate output. For seed 43 use
the `new-views-fixed-teacher-seed43-20261008` parent and its corresponding scene
report. For the original panel use `--panel original` and the original report.

## Checks and next direction

26 new tests cover group/source/hash separation, bounded configuration,
masked quality loss, frozen-head trainability and stale-suppression rejection.
Full repository suite: **485 passed, 2 skipped**, with six existing TensorFlow
export/interpreter warnings. Scoped Ruff passes.

Stop treating this scalar quality head as an established improvement. The next
useful experiment should target proposal geometry and blue/clutter discrimination
using permissible training-only sources, with a fixed control and the same
evidence boundaries. More epochs or repeated tuning on the tiny development
panel would not supply independent proof. Any eventual candidate still requires
isolated original-Pi timing and representative, independently labeled footage.
