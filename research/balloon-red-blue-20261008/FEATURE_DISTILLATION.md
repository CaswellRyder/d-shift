# Paired teacher-feature distillation

2026-10-08. **Do not promote this experiment.** Feature matching reduced indoor
blue recall in both seeds. The seed42 precision improvement did not repeat in
seed43. Existing Pi weights and deployment settings remain unchanged. This is
completed research, not the 94% flight-readiness gate.

## Hypothesis and training

Transfer the frozen expanded teacher's penultimate features, in addition to
its logits, without enlarging the exported student. The teacher has 1,280
features; the separable-context student has 128. A training-only Dense projection
maps between them and is discarded before export.

Four runs: seeds 42 and 43, each with a zero-weight control and feature-loss
weight 1. Both arms include the projection during training, start from identical
student weights within a seed, and use the same NumPy-shuffled batches. The
reviewer verifies paired provenance, including initial weights, target-cache
identity, data order, training sources, and configuration.

- Frozen manifest: `data/balloon-red-blue-expanded-views-20261008/manifest.json`;
  SHA `6011ae406f036759d9e6e4b334b5dd73de82b61248160d471c409ffede7e4cbb`.
- Frozen teacher: `runs/balloon-red-blue-teacher-expanded-20261008/teacher.keras`;
  SHA `db57446ecfba29c29353924792908d9a762f5452f50664f66c43cfdcc0ff45fe`.
- 584 approved real training entries; eight validation crops. Related views are
  not independent scenes. Entire IMG family remains development-only.
- Teacher uses 96x96 RGB crops; student uses 64x64 RGB crops from the same source.
  No augmentation in this paired experiment.
- Adam 0.001, batch 16, 25 epochs. Checkpoint chosen by minimum validation
  hard-label cross entropy, not by development-scene scores.
- Base loss: `0.5 * hard_CE + 0.5 * KL(teacher_T4 || student_T4) * 16`.
- Intervention adds mean cosine distance between normalized projected student
  features and normalized frozen teacher features. Targets are cached for
  training rows only; the reviewer checks cached row order against the manifest.

The new control uses a custom training loop/shuffle. Comparisons against old
historical student runs are not an isolated test of feature matching; use the
new matched controls for that claim. The training input guard was imported from
the existing working tree, not modified in this experiment; its hash is retained
in each provenance record.

## Training outcomes and export

- Control42: best epoch25, validation CE 0.006279; final hint loss 1.022670.
- Hint42: best epoch22, validation CE 0.000759; final hint loss 0.208018.
- Control43: best epoch20, validation CE 0.000240; final hint loss 1.010911.
- Hint43: best epoch18, validation CE 0.000181; final hint loss 0.214008.

Control hint loss is unoptimized by design. Lower validation CE on eight crops
and lower training feature loss did not establish better full-scene detection.

Each selected student has **7,763 parameters**. The 165,120-parameter training
projection is not in the pupil model. FP32 files are 35,204 bytes; INT8 files are
15,656 bytes. The reviewer allocated every TFLite export and verified identical
operator connectivity, tensor shapes, and dtypes across all four arms within
each export type, with a 1x64x64x3 input and 1x3 output. This establishes unchanged
export topology, **not a new measured Pi FPS**. No new model ran on the Pi.

INT8 calibration used training data only. Validation/test boundaries remain
unchanged; reserved test images were not evaluated or used for calibration.

## Full-scene results

Unchanged confidence threshold 0.8, class-aware one-to-one IoU 0.5 matching,
MSER proposals, duplicate suppression and confirmed-parent part suppression.
The indoor panel contains 12 reviewed scenes (11 red and 12 blue target boxes,
including three negative frames). Original panel: four reviewed off-domain
photos (two red, three blue). These panels have been reused for development and
are not an independent flight test.

Indoor FP32 counts below are **TP / FP / FN**:

- Control42: red **11 / 1 / 0**, blue **9 / 5 / 3**.
- Hint42: red **11 / 1 / 0**, blue **8 / 2 / 4**.
- Control43: red **9 / 2 / 2**, blue **8 / 3 / 4**.
- Hint43: red **10 / 3 / 1**, blue **7 / 4 / 5**.

Thus seed42 blue precision improves 64.3% to 80.0%, but recall falls 75.0% to
66.7%. Seed43 blue precision falls 72.7% to 63.6% and recall falls 66.7% to 58.3%.
Red seed43 gains one true positive but also one false positive. All four INT8
exports have the same indoor aggregate counts as their FP32 versions; this is
not bitwise prediction parity.

On the original panel, all FP32 arms have red 2 / 2 / 0. Blue is 3 / 0 / 0 for
both controls and hint42, but 3 / 1 / 0 for hint43. Complete FP32/INT8 results
and other diagnostic proposal variants are retained in the raw scene reports;
those extra variants were not used to select a new operating policy.

## Evidence, checks and next work

Full current-working-tree suite: **655 passed, 2 skipped**. Scoped Ruff checks
passed. New tests cover training-family exclusion, finite feature normalization,
cache identity, projection exclusion, and rejection of mismatched paired inputs.
The suite includes existing unrelated working-tree edits; it is not evidence
that those edits belong to or were reviewed as part of this experiment.

- `feature-distillation-summary.json`: paired provenance, source/weight/report
  hashes, model graph contracts, and compact scores.
- `feature-distillation-indoor-scenes.json`: eight exports, all scene detections.
- `feature-distillation-original-scenes.json`: eight exports, crop and scene
  diagnostics; negative-training-fit checks are not independent validation.
- Ignored `runs/balloon-feature-{control,hint}-seed{42,43}-20261008` contain
  training history, checkpoints and exports. Cached targets remain ignored in
  `artifacts/balloon-feature-targets-20261008`.

Reproduce each training arm with a fresh output directory:

```sh
TF_CPP_MIN_LOG_LEVEL=2 .venv/bin/python -m scripts.distill_balloon_features --manifest data/balloon-red-blue-expanded-views-20261008/manifest.json --teacher runs/balloon-red-blue-teacher-expanded-20261008/teacher.keras --cache artifacts/balloon-feature-targets-20261008 --output runs/fresh-feature-hint42 --seed 42 --feature-weight 1 --epochs 25
```

Set weight0 for its control and repeat at seed43. Export FP32 with
`scripts/export_pi_float.py`; evaluate using the existing original and indoor
scene evaluators. `scripts/review_balloon_feature_experiment.py` checks the
canonical experiment directories and writes a new, never-overwritten summary.

Next: preserve the existing candidate and profile the exact search path using
fixed inputs. Further accuracy work must address blue proposal coverage and
clutter discrimination, rather than treating eight-crop validation loss as a
substitute for full-scene performance. No motor, ESP32 or network configuration
commands were issued for this experiment.
