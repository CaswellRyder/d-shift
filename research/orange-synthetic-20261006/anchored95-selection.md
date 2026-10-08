# Fixed development candidate — October 7, 2026

Chosen before running the development comparison: gap9 supplement, maximum two
new orange candidates after baseline twelve, supplemental score >=0.95, retain
baseline decisions, reject additions with IoU >0.3 or intersection/smaller-area
>0.6 against accepted baseline or previously admitted supplemental detections.
Baseline acceptance remains 0.8. These scores are not calibrated probabilities.

Training screen of 369 images: baseline orange TP146/FP107; anchored95 TP159/FP110.
Anchored90 TP160/FP112. Select95 for fewer extra false positives, not by looking
at validation results. Same context FP32 weights and nested baseline suppression.

Next input is the existing 595-frame development_validation manifest. It has
been used elsewhere in development and is NOT an untouched independent test set.
Only baseline and fixed anchored95 will be evaluated. No sweep on that set.
This is stateless testing on Mac; no live timing or deployment approval follows.
