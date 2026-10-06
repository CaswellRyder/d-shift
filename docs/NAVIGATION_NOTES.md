# Planned navigation behavior

## Opposing-goal recovery — requested 2026-10-06

**Status: implementation note only; not implemented or flight-tested.** No model,
runtime policy, controller connection, or motor behavior changes with this note.

User requirement: if our assigned-color goal is lost or cannot be found, locate
the opposing team's goals and search in the opposite direction, away from them.

Planned behavior:

1. During delivery, enter goal-recovery after losing our goal or failing to find
   one for a configurable interval. Delivery remains eligible only when balloon
   possession is confirmed; otherwise stay in balloon-search/capture logic.
2. Use positively identified, fresh opposing-color goal observations as a
   directional reference. Unknown color is not evidence of an opposing goal.
   Keep opposing goals observable even though they are not scoring targets.
3. Bias a bounded search away from the opposing goals, continuing to scan for
   our assigned color. Use camera-bearing/IMU alignment when available; do not
   interpret image-left/image-right as a world heading without that alignment.
4. Exit recovery when our valid goal is reacquired. If the reference becomes
   stale, opposing-goal bearings conflict, or the recovery time/distance budget
   expires, fall back to the configured safe search/hold behavior. Do not keep
   travelling indefinitely on a cached opposing-goal bearing.
5. Require positive assigned-color goal confirmation and normal approach checks
   before delivery; the inferred search direction does not authorize goal entry.

Geometry prerequisite: confirm how opposing and assigned goals are arranged in
the actual arena. A goal bearing alone does not establish that our goal is exactly
180 degrees away, or that the intervening path is clear. Treat “opposite direction”
as a search heuristic, not an inferred destination or verified free-space route.
Navigation must retain arena-boundary, obstacle, and controller safety limits.

Before enabling: test lost/not-yet-found goals, multiple opposing goals, uncertain
color, stale observations, reacquisition, timeout, and absent/unknown balloon
possession. Recovery thresholds and field geometry remain to be specified; this
does not block continued CV training or optimization.
