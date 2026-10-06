# Close-range training exposure experiment — 2026-10-05

Status: epoch 1 saved/evaluated; epoch 2 saved. Experiment paused while prioritizing
the user's Pi integration and two-method comparison. No automatic continuation.
Verify process state before resuming; never start overlapping writers.

## First development result: yellow-square gain, other-class regressions

`interim-a/development-640.json`: all 595 unchanged validation frames, CPU/640,
confidence 0.25, same-class one-to-one IoU 0.5.

| Class | Precision | Recall |
| --- | ---: | ---: |
| Orange circle | 95.14% | 94.00% |
| Orange square | 92.45% | 96.08% |
| Orange triangle | 77.11% | 95.52% |
| Yellow circle | 85.00% | 96.23% |
| Yellow square | 93.20% | 91.37% |
| Yellow triangle | 98.33% | 99.16% |

**FAIL:** orange-triangle and yellow-circle precision. Yellow-square recall rises from
228/255 to 233/255, but orange-triangle false positives rise from 29 to 57 and yellow-circle
false positives from 5 to 27 versus the initialization checkpoint. The complete model ranks
below refinement epoch 2 in `comparison-through-epoch1.json`; no model is promoted.

Saved model SHA256: `93a4c2c0d65bf33888d517a39aa4b0ddac26856b7e322f30bd0dba54c601debd`.
Report SHA256: `87095d39cba869eb455f856b1cb45d7ef42b0ff6e30f8a66cc88b1590d6cd145`.
The epoch-2 resume verified exact full-precision training-state and runtime-profile restoration,
with RNG sidecar SHA256 `14841581337d552347279ee248e9002d71941ba52a150b7c18e5a6331092ea6d`.

Cached diagnostics (`interim-a/errors-640.json`) and reviewed contact sheets (`interim-a/review/`):

- 25 of 27 yellow-circle false positives overlap labeled yellow squares at IoU >= 0.5;
  one overlaps a yellow-triangle annotation, and one has no labeled overlap. The reviewed
  square crops visibly show square goals. One apparent circle under a triangle annotation
  warrants a separate annotation audit, not an automatic score correction.
- Orange-triangle false positives: 27 same-class localization overlaps below IoU 0.5,
  26 without labeled overlap at IoU 0.1, and four duplicates at matching IoU.
  The top-confidence review again contains partial triangle views. These overlap categories
  are descriptive, not proof of missing/wrong annotations.

The first checkpoint shows a tradeoff, not resolution of the six-class objective. Keep the
existing four-epoch cap and fixed criteria. The private web viewer remains on its previous
MobileNet pipeline; this checkpoint is not substituted there.

## Reason and bounded change

The [error review](GOAL_ERROR_FOLLOWUP.md) found that 26 of 27 yellow-square misses in the
strongest previous checkpoint share one close-up filename group. Only 25 training boxes
span more than half the frame, all from another group. Orange-triangle false positives
also include close, partially visible goals and below-threshold box overlaps.

This experiment repeats each training frame containing **any** goal whose normalized longest
box side exceeds 0.25, for four total copies. The same rule applies to all six classes.
Every original frame, including negatives, remains. Each repeat retains all labels in its
source frame, not just the large goal. No pixels or boxes are synthesized or relabeled.

- Unique training frames: unchanged at 2,950.
- Selected training frames: 349; three additional copies each.
- Effective training entries: 3,997 (previously 2,950).
- Qualifying boxes: orange circle 34, orange square 102, orange triangle 124,
  yellow circle 38, yellow square 26, yellow triangle 25.
- Validation: the same 595 images and original labels; receipt unchanged exactly.
- Reserved test: not read, exported or evaluated.

Repetition increases exposure, not independent data or background diversity. It can overfit
the same scenes. Changing initialization, optimizer schedule and exposure together also means
this is not a controlled causal estimate of repetition alone.

## Artifacts and initialization

Dataset: `data/goal-detector-close-exposure-20261005`.
Receipt SHA256: `7753900e9182bb881f3744e4bdbdc3f16a7ecb4cd8a75b2d9db40aae612ea759`.
Parent dataset: `data/goal-detector-v10-reviewed-color-20261005`.
Source image and label hashes were verified before export and after linking/copying.
Export refuses cross-split duplicate images, test-containing sources, source tampering,
repeated exposure derivatives, existing outputs and invalid label rows.

Run: `runs/goal-detector-close-exposure-20261005`.
Initialization: `runs/goal-detector-refined-20261005/interim-b/model.pt` (refinement epoch 2).
SHA256: `69a694e2e5acfb666fe679288173499fd8ed44eed7c50c0017bcb00badc79bce`.
The captured pre-forward model matches all 499 parent EMA state entries exactly after FP32
conversion. Runtime training profile verification passed; these are engineering checks only.

Fresh AdamW, learning rate 0.0001, lrf 0.01, no warmup/mosaic, four-epoch cap,
640 pixels, batch 8, MPS, workers 0, seed 42. Remaining refinement augmentation stays unchanged.
One saved epoch per process retains the existing native-memory containment and state restore.
The completed original and 12-epoch refinement experiments are not resumed or modified.

## Reproduce and continue

From the repository root, create the dataset once:

```sh
PYTHONPATH=src .detector-venv/bin/python scripts/prepare_close_range_detector.py \
  --source data/goal-detector-v10-reviewed-color-20261005 \
  --output data/goal-detector-close-exposure-20261005 --threshold 0.25 --copies 4
```

First launch (already performed; do not repeat):

```sh
PYTHONPATH=src .detector-venv/bin/python scripts/train_goal_detector.py \
  --data data/goal-detector-close-exposure-20261005 \
  --output runs/goal-detector-close-exposure-20261005 --epochs 4 --batch 8 \
  --refine-from runs/goal-detector-refined-20261005/interim-b --max-process-epochs 1
```

At a verified terminal saved boundary (exit 75), capture and evaluate a new immutable snapshot
before resuming. The resume command replaces `--refine-from ...` with `--resume` and retains
the same data, output, four-epoch cap, batch and one-process-epoch limit.

Compare each whole checkpoint with refinement epoch 2 on all 595 unchanged validation
frames at CPU/640, confidence 0.25 and matching IoU 0.5. Report every class, not just close-up
yellow squares. A gain in one class cannot offset another failing class. No promotion,
distillation, viewer replacement or Pi/flight qualification follows automatically.

## Engineering verification

Full suite: 140 passed, two environment-specific NMS tests skipped, four existing TFLite
warnings. Ruff passed. Five new sampling tests cover preserved validation/all labels and
backgrounds, existing output refusal, recursive repetition refusal, label tampering,
cross-split leakage, test scope, unsafe names, invalid rows and thresholds. Software test
results do not establish detector accuracy.
