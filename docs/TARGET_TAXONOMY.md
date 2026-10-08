# Current competition targets

User clarification on 2026-10-07 supersedes prior green/purple target assumptions:

- Balloons: red and blue only.
- Goals: orange and yellow, with circle/square/triangle shapes.

This records the user's rule clarification, not a new independent rulebook audit.

## Migration not yet complete

`configs/balloon-red-blue.json` defines the corrected training classes. It is not
a deployed model. The historical `configs/balloon.json`, reviewed datasets,
cached teacher targets, model metadata, and historical runtime profiles describe
the older green/purple task and remain unchanged for reproducibility. They must
not be represented as competition-ready red/blue balloon detection.

The additive `balloon_red_blue` search profile now includes red hue wraparound,
blue proposals, connected-component hole rejection, and the existing candidate
budget. These HSV bands propose regions, not verified balloon identities; orange
goals and other colored objects still require classifier rejection. Static,
detail, and temporal observation reject legacy weights with this profile and
reject red/blue weights with historical profiles. The webcam service selects the
profile from red/blue model metadata without changing its goal search profile.
Existing default models are NOT red/blue models and are not automatically replaced.

Next balloon work must review red/blue source labels, retrain the teacher/student or a suitable candidate,
and evaluate color confusion plus Pi performance. Never rename green/purple
logits or map old labels to red/blue as a shortcut. This correction does not
invalidate orange/yellow goal measurements, which are a separate task.

Code verification uses generated color patches and mock classifiers to check mask
boundaries, budgets, identity routing, and mismatch rejection. It establishes no
real balloon accuracy, Pi speed, or trained red/blue model availability.

The existing generated foil scene actually contains green/purple balloons. Keep
those image annotations truthful; it remains a goal-negative research scene,
not red/blue positive training or evidence about competition balloon behavior.
