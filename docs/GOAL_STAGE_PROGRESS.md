# Goal failure audit — 2026-10-05

## Decision

**Keep the selected viewer unchanged:** `balloon_components`, 12 candidates per task,
proposal-adapted teachers, threshold 0.8, nested duplicate suppression, 320x240 input.
No new weights, real-data distillation, test inference, camera access or Pi changes.

This follow-up adds a reproducible stage-level diagnostic, not a new accuracy claim.
It retains every candidate's class scores, accepted/suppressed state, box and matched
annotation. It distinguishes missing geometry, candidate-budget loss, background,
wrong class, confidence rejection, suppression and one-to-one assignment conflicts.
False positives are separately partitioned into wrong class, partial box, duplicates
and no labeled overlap. These are annotation-relative categories, not corrected labels:
upstream omissions/conflicts remain possible.

## What actually blocks orange goals

The audit exactly reproduces all six goal-class TP, FP and target totals in the selected
595-frame development-validation receipt, with identical teacher, annotations and vision
source hashes. Goal precision/recall remains **72.3%/54.6%**; balloon remains **77.8%/81.8%**.

| Stage outcome | All goals | Orange goals only |
|---|---:|---:|
| Detected | 649 | 168 |
| No IoU>=0.5 box, even with 64 diagnostic proposals | 328 | 325 |
| Box exists at diagnostic budget but absent at 12 | 125 | 99 |
| Covered, but wrong non-background class | 49 | 32 |
| Correct top class, below threshold | 26 | 22 |
| All covering crops classified background | 9 | 9 |
| Correct accepted crop removed by duplicate suppression | 2 | 0 |
| Total labeled targets | 1,188 | 655 |

Each target appears once, using the furthest successful stage. Diagnostic 64-candidate
boxes receive **no teacher inference**; this is not a prediction of performance at budget 64.
Mixed causes can exist across multiple crops. The partition is triage, not a causal study.

Of 325 orange localization failures, **318 have a longest bounding-box side under 16 pixels**
at 320x240. Tiny orange targets are more prevalent in validation (503/655) than in the fixed
training audit (88/208). This describes the sampled distributions, not independent recording
quality. The classifier cannot recover an object for which it never receives a useful crop.

## Experiments retained, not promoted

1. **`goal_gap9`:** adds a 9x9 closing pass to existing raw/5x5 goal masks to reconnect broken
   rims. HSV ranges, models, threshold and candidate cap remain unchanged. Balloon uses the
   selected component filter. Orange train coverage improves **151/208 to 161/208**, but full
   validation goal detections fall **649 to 578**, false positives rise **249 to 269**, and F1
   falls **62.2% to 56.8%**. Yellow-triangle true detections fall from 105 to 47. Rejected.
   This profile is audit-only; the viewer does not expose it.
2. **640x480 geometry audit:** original selected proposal logic at higher resolution gives
   orange train coverage **141/208**, versus 151/208 at 320x240. No full-frame validation or
   teacher inference was run for this variant. The existing pixel-based area thresholds and
   kernels were deliberately unchanged; this does **not** isolate resolution independently
   of their relative scale, and does not prove higher-resolution strategies cannot work.
   Viewer resolution stays unchanged. There is no Pi timing or memory qualification.

The next useful experiment should address small-target localization and ranking explicitly,
with a size-stratified train audit, before more classifier training. Avoid another blanket
mask expansion or larger candidate budget without measuring added clutter and runtime cost.
Separately recorded Pi-camera footage remains necessary for qualification.

## Reproduce

From the project root, choose fresh output paths:

```bash
.venv/bin/python scripts/diagnose_goal_frames.py \
  --split train --stride 8 --output /tmp/dtr-goal-train-stages
.venv/bin/python scripts/diagnose_goal_frames.py \
  --split valid --output /tmp/dtr-goal-valid-stages
.venv/bin/python scripts/audit_goal_proposals.py \
  --profile goal_gap9 --output /tmp/dtr-gap9-train
.venv/bin/python scripts/audit_goal_proposals.py \
  --profile balloon_components --width 640 --output /tmp/dtr-resolution640-train
make test
```

The diagnostic CLI permits only training/development validation, never the reserved test.
Reports are marked `training_approved: false`; validation diagnostic crops/scores must not
become training samples. Original dataset annotations and manifests are unchanged.

Retained evidence:

- `data/goal-stage-{train,valid}-20261005/`: complete reports plus exact diagnostic/vision
  source snapshots; 369 fixed training frames and all 595 development-validation frames.
- `data/proposal-audit-gap9-20261005/`: train audit, validation, source snapshot, comparison.
- `data/proposal-audit-resolution640-20261005/`: train geometry audit and missed-target sheets.

Verification: Ruff and **57 Python tests passed** (four existing TFLite warnings). Tests cover
each diagnostic stage, one-to-one assignment, duplicate/wrong-class false positives, partial
boxes, and unchanged balloon behavior in the experimental profile. No viewer/UI code changed
or browser QA claimed in this follow-up. Existing loopback viewer remains on port 8765.
