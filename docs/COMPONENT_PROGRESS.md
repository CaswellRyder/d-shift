# Component-filter follow-up — 2026-10-05

## Selected change

Viewer defaults now use **`balloon_components`**, both 12-candidate budgets, the same
proposal-adapted teachers, threshold 0.8 and existing duplicate suppression. Goal proposal
behavior is unchanged. No new model training, distillation, test inference or Pi deployment.

| Task | Precision before → after | Recall before → after | F1 before → after | False detections before → after |
|---|---:|---:|---:|---:|
| Balloon | 44.4% → 77.8% | 81.8% → 81.8% | 57.6% → 79.8% | 663 → 151 |
| Goal | 72.3% → 72.3% | 54.6% → 54.6% | 62.2% → 62.2% | 249 → 249 |

Same 595 development-validation frames, teacher checksums and annotations. Balloon true
detections remain 530: green 427/541, purple 103/107. False positives are green 92 and purple 59.
Independent recordings, camera performance, tracking and flight qualification remain absent.
Repeated development on this validation block means these are model-selection measurements,
not final generalization estimates. Existing upstream label problems remain relevant.

Visual error triage showed many high-confidence unmatched balloon predictions covering only
small reflective regions of an actual balloon. The earlier `RETR_LIST` contour enumeration
treated holes in a foreground component as additional object candidates. The selected profile
uses `RETR_CCOMP` and rejects child/hole contours, while preserving foreground islands inside
hoops. It does not expand or invent bounding boxes or alter learned classifier scores.

Plain `RETR_EXTERNAL` was also measured but **not selected**: balloon true detections fell to
483, because removing all nested contours can erase real foreground objects. The component
version retains the original 530 detections. A regression fixture explicitly verifies that
a small balloon inside a same-color hoop survives while hole contours are excluded. This is
an engineering edge-case check, not a proof for every real occlusion.

## Orange-goal work: retained failures, not default changes

On the fixed train-frame audit, baseline orange coverage is 151/208 targets at IoU>=0.5.
Early aggressive rim consolidation (`rim_v4`) falls to 137; conservative consolidation
(`rim_budget`) ties at 151; separate local-red-contrast proposals (`orange_local`) fall to 92.
These experiments remain audit-only opt-ins, not viewer profiles. No validation inference
was used for these three rejected same-budget variants. Wider HSV experiments from the prior
step also remain rejected. Missing orange goals are **not solved** by this follow-up.

## Optional higher-budget desktop diagnosis

Increasing only the goal candidate budget to 24 improves train orange coverage to 171/208.
On the same full validation frames, goal recall rises from 54.6% to **59.4%**, but precision
falls from 72.3% to **62.0%** and F1 falls from 62.2% to **60.7%**. True detections rise from
649 to 706 and false detections from 249 to 433. Therefore **12 remains the default**.
This is an unequal-candidate-budget comparison, not a free accuracy or Pi performance gain.
The earlier 24-candidate receipt uses `v2`; the selected component profile changes balloons
only, and goal behavior is identical.

For deliberate desktop comparison, stop the usual viewer first or use a separate port:

```bash
.venv/bin/python webcam_app.py --allow-unvalidated --goal-proposal-limit 24 --port 8766
```

The viewer displays the actual task budget and a higher-compute warning. Balloon stays at 12.
`--proposal-profile v2` restores the old contour behavior for controlled comparisons. Failed
orange/rim variants cannot be selected through the viewer CLI. General library and historical
audit defaults remain `v2`; pass explicit profiles when reproducing selected behavior.

## Reproduction and retained evidence

```bash
.venv/bin/python scripts/evaluate_dtr_frames.py \
  --balloon-model runs/balloon-proposals-20261005/teacher.keras \
  --goal-model runs/goal-proposals-20261005/teacher.keras \
  --profile balloon_components --duplicate-policy nested --goal-limit 12 \
  --output /tmp/dtr-components.json
.venv/bin/python scripts/compare_dtr_frames.py \
  --before data/proposal-audit-20261005/full-frame-final.json \
  --after /tmp/dtr-components.json --allow-vision-change \
  --output /tmp/dtr-components-comparison.json
```

Use fresh output paths. Budget comparisons additionally require `--allow-budget-change` and
fixed teacher hashes; the receipt explicitly records each task's before/after budget.
Equal proposal budgets are not automatically equal total compute or latency.

- `data/balloon-components-20261005/{report,full-frame,comparison}.json`: train audit and validation.
- `data/balloon-components-20261005/full-frame.vision.py`: exact evaluated implementation snapshot.
- `data/balloon-contour-{baseline,external}-20261005/`: diagnostic baselines.
- `data/proposal-audit-{rim-v4,rim-budget,orange-local,budget24}-20261005/`: other attempts.
- `data/detection-review-20261005/`: per-class missed/unmatched contact sheets plus source indices.
  These use the **pre-component validation receipt** to document why this change was made.
  They are diagnostic examples, not corrections or new training labels. No validation images
  were moved into training. Confidence-ranked samples are not representative prevalence estimates.

Local timing remains uncontrolled and includes warmup; no speed gain or ARMv6 claim follows.
Next priorities: orange-goal localization/class confusion, remaining balloon errors, then
separately recorded Pi-camera sessions before distillation or independent final evaluation.

## Verification

Ruff, JavaScript syntax checks and **52 Python tests passed**; four existing TFLite warnings.
The evaluated vision source matches the retained source snapshot/checksum. Playwright checked
the selected component profile and teacher hashes in both 12- and 24-goal-budget modes,
saved-image analysis for both tasks, duplicate explanations, Stop and mobile reanalysis.
No console errors, viewport overflow or camera activation; screenshots were visually reviewed.
The skill's CLI wrapper was unavailable, so the already-installed browser library was used.

- Default-mode browser receipt/screenshots: `output/playwright/viewer-1791180083588/`.
- Optional 24-goal-mode receipt/screenshots: `output/playwright/viewer-1791180086050/`.

Only the default 12-candidate viewer was left running on port 8765. The temporary port-8766
QA server was stopped; use the command above to start that opt-in mode yourself. Camera
permission remains user-controlled. No Pi mutations, laptop sync, reserved-test inference,
distillation, model-weight updates or flight commands were performed in this follow-up.
