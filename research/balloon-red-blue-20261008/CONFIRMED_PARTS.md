# Confirmed-parent suppression experiment

Research only, 2026-10-08. No production/default proposal or suppression policy
changed. All results are four reused, reviewed, off-domain development photos
containing only two red and three blue target balloons, not the 94% gate.

Earlier MSER containment filtering acted **before** classification. That let a
large unverified region hide a real smaller balloon. The new experiment leaves
the same 12 MSER proposals intact, classifies them all, applies existing duplicate
suppression, and only then suppresses small parts inside accepted same-color
balloon parents. The child must be <=35% of the parent bounding-box area and
>=95% contained. Background, rejected, differently colored, and goal parents
cannot suppress a balloon. Original prediction records are preserved.

| Context / separable model, same MSER search | Red TP / FP / FN | Blue TP / FP / FN |
| --- | ---: | ---: |
| Existing duplicate suppression | 2 / 9 / 0 | 3 / 0 / 0 |
| Pre-classification containment experiment | 2 / 2 / 0 | 1 / 0 / 2 |
| New confirmed-parent suppression | 2 / 2 / 0 | 3 / 0 / 0 |

The refined-context candidate still misses two blue targets under either MSER
post-processing policy. The new rule cannot restore targets rejected by the
classifier. Red precision is still only 50% for confirmed-parent MSER, and the
remaining red false positives are in the cluttered people/cone photo.

This supports further investigation, not deployment: two same-color balloons
overlapping in projection can still be merged incorrectly, and an incorrectly
accepted parent can still suppress a true child. Test those cases explicitly on
new representative recordings. No Pi MSER timing claim is made here; the separate
native-mask experiment accelerates the baseline HSV proposal path, not MSER.

Evidence: `confirmed-parts-development.json` contains hashes, per-scene ground
truth, all detections and suppression reasons for context, refined and separable.
No reserved-test photos were opened. No thresholds were tuned or weights retrained
for this experiment. Unit fixtures check parent acceptance, class agreement,
containment, unchanged input evidence, and isolation from goal detections.
