# Open-method research plan — October 6, 2026

User explicitly welcomes synthetic imagery and alternative methods. The prior
recommendation to avoid architecture changes is not a project constraint.
Preserve the integration baseline while conducting separate, measurable research.
ESP32 attachment remains out of scope.

## First synthetic diagnostic

`research/orange-synthetic-20261006/` contains one generated, visually inspected
gym scene, approximate annotations, exact prompt, and proposal-coverage receipts.
It includes orange balloons, an orange square goal, a yellow circular goal, and
orange cones. This is one research seed, not an adequate training dataset.

At 320x240 with a twelve-candidate limit, baseline goal proposals failed to cover
the orange square at IoU >=0.5 (best IoU about 0.039). The existing experimental
`orange_local` method covered it at about 0.753 IoU. Baseline yellow circle coverage
was about 0.791. These are proposal boxes, NOT neural classification results.
No speed advantage or real-image improvement is established.

Increasing baseline candidates to 64 did not repair this orange miss. Separately,
`goal_gap9` covered it at 64 candidates but not 12; this suggests both proposal
generation and ranking deserve separate ablations. Do not adopt a 64-crop live
inference budget on that evidence.

Current balloon masks and label config support green/purple, not orange. Orange
balloons in this generated scene are not covered. That is an unsupported task,
not measured classifier failure. Their required target/negative role awaits user
confirmation; leave production labels and weights unchanged meanwhile.

## Research tracks, in priority order

1. Proposal coverage: compare baseline, bounded local orange-contrast search,
   and size-aware candidate ranking. Hold the classifier fixed to isolate gains.
2. Compute scheduling: compare bounded periodic wider scans and selective crop
   verification; preserve unknown/lost states and stale-label expiration.
3. Training data: independently generated scenes with scale, exposure, blur,
   partial occlusion, rim/body confusion, and warm-colored distractors. Mix with
   reviewed real training data; keep source groups intact and holdouts real.
4. Learned alternatives: test a color-agnostic object-type head plus a separate
   color estimate, or a smaller multi-head student. Changes to heads/labels require
   new training and export; don't relabel an existing model at inference time.

For each candidate record proposal recall, end-to-end precision/recall by color
and apparent size, any-target success, reacquisition, skipped frames, and Pi
p95 result age, FPS, and memory. Tune on development data only. Preserve an
independent real holdout for final selection. Compare one method per Pi process.
Synthetic variants from the same scene must never cross data splits.

## State at handoff

- Baseline model, deployment thresholds, and Pi files unchanged this turn.
- Generated asset saved in the repository, not just the tool output directory.
- Probe runs at 320x240 and 640x480; the existing edge-search implementation is
  explicitly skipped at 640x480 because it only supports at most 320x240.
- Twenty-nine existing proposal/temporal tests passed; probe lint passed.
- No retraining performed; no new Pi timing measured. Pi was unreachable during
  the preceding session. The orange-local candidate needs real-data false-positive
  evaluation and Pi profiling before any baseline replacement.
