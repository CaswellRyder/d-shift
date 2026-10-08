# Current competition targets

User clarification on 2026-10-07 supersedes prior green/purple target assumptions:

- Balloons: red and blue only.
- Goals: orange and yellow, with circle/square/triangle shapes.

This records the user's rule clarification, not a new independent rulebook audit.

## Migration not yet complete

`configs/balloon-red-blue.json` defines the corrected training classes. It is not
a deployed model. The historical `configs/balloon.json`, reviewed datasets,
cached teacher targets, model metadata, and runtime color masks still describe
the older green/purple task and remain unchanged for reproducibility. They must
not be represented as competition-ready red/blue balloon detection.

Next balloon work must review red/blue source labels, add red/blue proposal masks
(including red hue wraparound), retrain the teacher/student or a suitable candidate,
and evaluate color confusion plus Pi performance. Never rename green/purple
logits or map old labels to red/blue as a shortcut. This correction does not
invalidate orange/yellow goal measurements, which are a separate task.

The existing generated foil scene actually contains green/purple balloons. Keep
those image annotations truthful; it remains a goal-negative research scene,
not red/blue positive training or evidence about competition balloon behavior.
