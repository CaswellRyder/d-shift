# Shape evidence without forced color — 2026-10-05

The user reports recognizing goal shape at approximately 95 feet while color
remained uncertain. This is an observation to reproduce, not a verified operating
range. No recording, measured setup, or identification of the running viewer/model
was supplied with that observation. Pixel size is not a measured physical range.

## Implemented

`dtr.goal_evidence` aggregates existing seven-class scores without another neural
invocation or a changed checkpoint. For example, orange-circle 0.475 plus
yellow-circle 0.475 produces circle evidence 0.95, with **color unknown**.
Shape confidence is the unconditional sum across those two classes. Color scores
are conditional on that same shape, not summed across unrelated shapes. The 0.8
evidence threshold is provisional; scores are not calibrated probabilities.

Both native/TFLite `Predictor` and Keras `TeacherPredictor` now include
`goal_evidence` in goal results. Original label, score, joint-class acceptance,
duplicate suppression, model files, and temporal expiry remain unchanged.
The crop-based browser viewer now shows these fields separately. Strong shape
with unknown color gets a purple `shape — color ?` label and a **tentative** count,
not an accepted detection. Suppressed, lost, or stale cached observations cannot
be promoted to tentative. Stop clears the count. No extra model call is added.
The separate YOLO viewer is unchanged. The existing crop viewer was restarted
with its same teachers and proposal settings; no experimental student replaced it.
See [CV progress](CV_PROGRESS.md) for the search and speed experiments.

`goal_candidate` provides an **opt-in observational investigation policy**:

| Evidence | Selected orange goal | Selected yellow goal |
| --- | --- | --- |
| Accepted matching shape/color | Confirmed model target | Confirmed model target |
| Strong opposite color | Reject | Reject |
| Strong shape, unknown color | Tentative only if explicitly enabled | Tentative only if explicitly enabled |
| Weak shape / background | Reject | Reject |
| Stale, missing age, lost or suppressed | Reject | Reject |

Unknown is never rewritten to orange. “Confirmed” here describes the model label,
not an independent truth or authorization to enter the goal. Every policy result
has `action_authorized: false` and `flight_commands: null`. It issues no pursuit,
capture, traversal, motor, or serial commands. Default investigation is disabled.
The policy requires supplied capture-derived measurement age (default maximum
500 ms), and rejects cached classification age over 1,000 ms. Those are provisional
research limits, not qualified blimp control deadlines. The current camera loop
does not yet supply a verified sensor-to-controller age contract.

This cannot recover an object omitted by the HSV candidate generator, nor prove
shape identity when the model itself lacks shape evidence. It exposes ambiguity
already present in the score vector. A real shape-only network head and a
color-independent candidate search would be separate trained/evaluated changes.

## Audit: keep the fallback experimental

Reused development data: 595 frames, context-distilled FP32 student, same saved
scores, threshold 0.8. No weights or thresholds were tuned in this audit.

| Orange candidate pool, correct shape and IoU >=0.5 | TP | FP | Recall |
| --- | ---: | ---: | ---: |
| Original accepted orange results | 210 | 245 | 32.06% |
| Plus tentative shape-known/color-unknown results | 210 | 256 | 32.06% |

The fallback added **11 false positives and zero additional true positives** in
aggregate. It is therefore **not enabled for autonomous pursuit**. This does not
resolve the user's different, unrecorded long-range scene. It establishes that
the available development set does not support promoting this fallback.

The separate color-agnostic shape audit also found no recall improvement (58.33%
before and after); precision changed 69.23% -> 68.07%. That looser task ignores
color and uses shape-based nested suppression, so it must not replace the existing
six-class accuracy results. Investigation-pool metrics are not confirmed detections.

Reproduction:

```bash
PYTHONPATH=src .venv/bin/python scripts/audit_goal_uncertainty.py \
  --observations runs/student-research-20261005/context-kd/frames-mac.frames.jsonl \
  --metadata runs/student-research-20261005/context-kd/student.float.json \
  --truth runs/pi-comparison-20261005/frames.json \
  --output runs/goal-uncertainty-repeat.json
```

Audit artifacts: `runs/goal-uncertainty-20261005/context-audit.json` and its
`.frames.jsonl` include evidence and selection annotations. The audit assumes
hypothetical zero measurement age for static images; this is not measured freshness.
`runtime-parity.json` is the new 595-frame inference replay used to check that
legacy acceptance metrics stay unchanged. `pi-crop-bundle/` is a new standalone
package including the helper; older frozen bundles remain intact. No Pi speed
claim is made for this change.

Verification: **210 tests passed, 2 skipped; Ruff passed.** All 6,664 observations
in the 595-frame fresh inference rerun retained exactly the old boxes, score
vectors, labels, acceptance and suppression fields, with the new evidence added.
The isolated bundle also passed all 21 local golden-crop comparisons (zero score
delta). These are local correctness checks, not live Pi or 95-foot range tests.

Next field evidence: save raw frames/video at measured distances with target
color/shape, lighting, camera settings, and actual running model/command. Include
yellow goals, orange goals, and non-goal distractors. Hold out complete recording
sessions. Keep unknown-color tracks observable while requiring positive target
confirmation before any final goal-entry decision.
