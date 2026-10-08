# Red/blue target and qualification contract

User request, 2026-10-08: retrain/retool and continue toward flight readiness at 94%.
Working interpretation: **precision AND recall >=94% for EACH of red and blue**,
not pooled accuracy or crop classification alone. This interpretation has been
communicated to the user. Preserve old goal and green/purple artifacts.

## Separate evidence levels

1. Engineering: tests, metadata/taxonomy guards, finite outputs, bounded budgets.
2. Crop development: teacher/student selection on source-disjoint development
   crops. Small generic-photo scores never satisfy competition accuracy.
3. Detection: one-to-one, same-class, IoU >=0.5 matches on held-out complete real
   scenes including negative scenes, near/far targets, blur and lighting changes.
   Count missed search proposals and stale/wrong tracks, not just classified crops.
4. Live original Pi Zero W: correct camera orientation, sequential isolated tests,
   sustained speed, p95 frame age/latency, dropped frames, memory and thermal state.
   Desktop timing or translated-image replay cannot substitute for live evidence.
5. Vehicle flight acceptance: physical capture/clearance, controller integration,
   safe-loss behavior and team-authorized flight tests. No ESP32/motor commands
   are authorized in this work; vision readiness is not whole-vehicle readiness.

## Evidence floor proposed for the perception gate

Use at least 100 labeled target instances per color across at least three
independent, representative held-out recording sessions, with full negative-scene
labels. Freeze thresholds and model/profile hashes before final evaluation.
Report counts, session-level results, size bins and confidence intervals alongside
the 94% point estimates. Adjacent video frames are correlated and do not create
independent trials. This is an engineering evidence floor, not a statistical
guarantee or a team-approved flight-safety specification.

The team's allowable end-to-end latency, range error, stale-track age and flight
clearance remain unspecified. Measure them; do not invent approval criteria or
mark deployment_approved true. Public training data and synthetic augmentation
remain useful for development while representative Pi evidence is collected.

## Immediate sequence

- Freeze a source-disjoint public-photo development split and tiny reserved
  holdout. Label their domain/sample-size limits prominently.
- Train a fresh MobileNetV4 teacher head with the corrected classes; do not
  rename legacy logits. Compare an inexpensive distilled student afterward.
- Diagnose full-frame search misses with bounded alternatives and hard-negative
  review. Keep training-derived selection apart from the final accuracy set.
- Export and test FP32/INT8 artifacts separately; preserve hashes and old weights.
- Run live Pi tests once SSH routing works, without concurrent competing methods.
- Keep commits scoped and record failed candidates as well as successful ones.
