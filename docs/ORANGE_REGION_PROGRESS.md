# Small-orange region experiment — 2026-10-05

## Decision

Both neon orange and neon yellow goals remain required. **Orange localization is not
resolved.** Keep the selected viewer on `balloon_components` and the existing teachers.
This experiment is retained for reproducibility but is not available in the viewer.

Implemented an orange-only MSER region source on red chroma, bounded to boxes with longest
side at most 32 pixels. Local contrast rejects flat warm backgrounds. It complements the
existing HSV contours without widening their mask. A second, compact variant suppresses
same-orange candidate overlaps above IoU 0.5 before inference; yellow retains 0.65.
Both variants preserve the selected balloon component path and the 12-candidate cap.
Neither region color nor convex-hull extent is an object identity or a goal aperture.

## Measured results

Fixed training sample: 369 frames, 208 orange targets, 149 yellow targets, IoU>=0.5.

| Proposal profile | Orange covered | Orange under 16px covered | Yellow covered |
|---|---:|---:|---:|
| Selected baseline | 151/208 | 46/88 | 124/149 |
| Regions, original candidate suppression | 151/208 | Not retained in initial audit | Not included in initial audit |
| Regions, compact orange suppression | 152/208 | 40/88 | 124/149 |

The aggregate +1 hid small-target and triangle regressions. The compact variant was also
evaluated on the same 595 development-validation frames, with fixed teacher hashes, threshold
0.8, frame resolution and candidate cap. Full-frame detections are one-to-one same-class matches:

| Metric | Selected baseline | Compact-region experiment |
|---|---:|---:|
| Goal true detections | 649 | 634 |
| Goal false detections | 249 | 252 |
| Goal precision | 72.3% | 71.6% |
| Goal recall | 54.6% | 53.4% |
| Goal F1 | 62.2% | 61.1% |
| Orange-circle true detections | 43 | 56 |
| Orange-square true detections | 53 | 52 |
| Orange-triangle true detections | 72 | 44 |

Yellow true detections rise 481 to 482, but yellow false detections also rise 140 to 146.
Balloon results are exactly unchanged (530 true, 151 false). The new approach does not qualify
for promotion. Same candidate cap is not equal total compute; region extraction adds work.
No ARMv6 speed/memory claim follows from local Mac inference.

## New training regression screen

The audit now records all goal colors, size strata, exact sampled frame names, source snapshots,
annotation checksum and IoU threshold. `compare_proposal_audits.py` requires matching training
scope, resolution and budget, all six classes and reconciled size counts. A candidate is
eligible for full validation only when orange coverage improves and **no class or class/size
stratum regresses**. Exit code 2 means rejection; exit code 0 only permits the next validation
step, never model or deployment approval. This is a conservative engineering screen, not a
significance test. It cannot guarantee generalization or detect every individual-frame tradeoff.

This screen was added after the full-frame experiment above. Replaying it rejects the compact
variant for orange-triangle loss and small-square/small-triangle loss. Future experiments should
run it **before** validation, avoiding this aggregate-only selection mistake.

```bash
# Run from the project root; choose fresh output paths.
.venv/bin/python scripts/audit_goal_proposals.py \
  --profile balloon_components --goal-colors all --output /tmp/dtr-region-baseline
.venv/bin/python scripts/audit_goal_proposals.py \
  --profile orange_regions_compact --goal-colors all --output /tmp/dtr-region-trial
.venv/bin/python scripts/compare_proposal_audits.py \
  --before /tmp/dtr-region-baseline/report.json \
  --after /tmp/dtr-region-trial/report.json --output /tmp/dtr-region-gate.json
```

Receipts are under `data/proposal-audit-region-gate-{baseline,trial}-20261005/`; the trial
directory includes `gate.json` with explicit rejection reasons. Validation and its comparison
are under `data/proposal-audit-regions-compact-20261005/`, including the evaluated vision source.
Earlier exploratory and size-audit receipts are retained separately, not overwritten.

Next localization work must recover small orange targets without displacing valid triangle
proposals. More crop-classifier training alone cannot repair proposal absence. The current
color/contour ranking remains a bottleneck; a learned proposal/ranking experiment would need
its own training-label review, full-frame comparison and device cost measurements.

No changes to teachers, dataset labels, viewer defaults, camera access or Pi state. Reserved
test data remains untouched. No real-data distillation. Tests cover faint-rim recovery, flat
background rejection, bounded proposals, unchanged balloon behavior and the new regression screen.
