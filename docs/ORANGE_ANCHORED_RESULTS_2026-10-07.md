# Baseline-preserving supplemental detections — October 7, 2026

## Outcome

A stricter admission policy improves aggregate orange detection in development,
without changing model weights or baseline detections. It is not deployed and has
not been timed on the Pi. Orange-circle precision still regresses.

Extra boxes previously displaced baseline boxes during nested duplicate
suppression. The revised policy first finalizes baseline decisions, then admits
only new orange detections at score >=0.95 that do not overlap or contain an
already accepted target. Supplemental overlap limits: IoU >0.3 or intersection /
smaller box area >0.6 causes rejection. Baseline threshold remains0.8. Scores are
not calibrated probabilities. Up to two extra candidate crops, no weight changes.

## Training selection

Tested cutoffs0.8,0.9,0.95 on the same369 training frames. Anchored95 produced
159 correct orange detections /110 unmatched, versus baseline146/107. It preserved
the baseline circle detections and all yellow output. The0.95 policy was recorded
before evaluating development; development did not sweep thresholds.

## Fixed development comparison

595 development images,655 orange and533 yellow annotated goals,320x240. Same
context FP32 model, stateless twelve baseline crops plus at most two supplementary
crops. The existing development set has been reused elsewhere; this is not a fresh
independent competition test. No reserved test data used.

| Orange metric | Baseline | Anchored95 |
|---|---:|---:|
| Correct detections | 210 | 231 |
| Unmatched detections (scored FP) | 245 | 259 |
| Recall | 32.1% | 35.3% |
| Precision | 46.2% | 47.1% |

| Class | Baseline TP / FP | Anchored95 TP / FP |
|---|---:|---:|
| Orange circle | 70 /92 | 71 /100 |
| Orange square | 47 /28 | 55 /29 |
| Orange triangle | 93 /125 | 105 /130 |

Yellow output is unchanged:482 TP,64 FP,51 FN. This uses same-color AND same-shape
matching; the earlier Pi comparison ignored shape for yellow localization, so
its484 TP is not the same metric. Do not splice these results into that table.

No existing accepted baseline box is removed, but supplementary conflicts can
still add false positives or prevent a better hypothesis from being admitted.
Orange circle gains only one correct detection for eight extra unmatched boxes.
Overall orange recall remains low; this is an incremental candidate, not a fix
for competition reliability. Incomplete upstream annotations also limit FP claims.

## Reviewed negatives

Visually inspected the three new training unmatched detections at cutoff0.95.
Two candidate goal hard negatives involve sign/packaging and a purple-foil-balloon
boundary. A narrow edge-clipped region stays ambiguous and is excluded. The
review file is `research/orange-synthetic-20261006/hard-negative-review-20261007.json`.
No samples were imported into training. Source training images must remain in
their original source groups for any descendants.

Real source balloons in these inspected scenes are reflective foil, unlike the
latex balloons in the first generated scene. Future synthetic assets should
include that material variation; generated imagery is not camera-domain evidence.

## Evidence and remaining work

- Training receipt: `data/proposal-audit-gap9-supplement-20261007/classification-anchored.json`.
- Development receipt: `data/proposal-audit-gap9-supplement-20261007/development-anchored95.json`.
- Prior selection: `research/orange-synthetic-20261006/anchored95-selection.md`.
- Evaluation script: `scripts/verify_orange_supplement.py`.
-48 relevant tests passed, lint passed. Baseline weights and deployment unchanged.

Execution used the Mac TensorFlow Lite CPU interpreter, not Pi timing. Live
four-crop scheduling still needs integration and measurement; no speedup is claimed.
Pi SSH/name resolution timed out again. ESP32 remains out of scope. Next work:
review broader training negatives, test their effect on a separate training run,
and profile a frozen candidate on Pi before changing the deployment baseline.
