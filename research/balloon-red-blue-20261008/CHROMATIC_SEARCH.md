# Brightness-normalized search — rejected for promotion

2026-10-08. Research only, unchanged threshold and twelve-candidate budget.
No model was retrained in this experiment; no Pi/runtime default changed.

## Hypothesis and training-only check

MSER currently searches positive red-minus-other and blue-minus-other intensity
planes. Normalize this contrast by the sum of RGB channels to recover darker
colored targets. Pixels with maximum channel below 30 are suppressed. This also
amplifies colored clutter; coverage alone cannot establish a benefit.

The algorithm was checked first on two training panels, with no IMG-family
images used for training, calibration, proposal selection or negative mining:

- Original public training targets: ordinary MSER covers 19/22 red and 7/15 blue;
  normalized MSER covers 20/22 red and 8/15 blue.
- Admitted external training targets: ordinary MSER covers 24/24 red and 20/24
  blue; normalized MSER covers 24/24 red and 21/24 blue.

Coverage means a same-color proposal has IoU >=0.5. It is not classifier recall
or precision; incomplete training-scene annotations cannot label other regions
as background. The new audit only uses explicitly admitted positive annotations
and validates archive, frame and source identities. The reserved IMG family is
rejected by the existing admission guard and checked again at frame loading.

Desktop original-panel search time rose from about 10.76 to 15.00 ms per frame
in this one-pass measurement. This is not a Pi speed estimate or sustained timing.

## Full-frame development falsifies the expected benefit

After the training-only coverage check, evaluated the frozen seed-42 and seed-43
fresh separable FP32 models on the twelve previously reviewed development scenes.
Both searches use identical confirmed-parent suppression and threshold 0.8.
The scene family remains development-only, not final qualification.

Seed 42, ordinary -> normalized MSER:

- Red TP/FP/FN: 11/2/0 -> 11/1/0.
- Blue TP/FP/FN: 8/2/4 -> 7/11/5.

Seed 43, ordinary -> normalized MSER:

- Red TP/FP/FN: 9/1/2 -> 8/2/3.
- Blue TP/FP/FN: 8/3/4 -> 8/13/4.

Decision: reject as a runtime replacement. Small training coverage improvements
do not survive the extra clutter and changed crops. The red improvement for one
seed does not offset the blue precision collapse. No threshold was tuned on these
results, no IMG examples were admitted into training, and no final-test predictions
were made. The 94% objective is not met.

## Artifacts and next work

- `chromatic-original-training.json`: all original training matches and timing.
- `chromatic-indoor-training.json`: admitted external training matches.
- `chromatic-indoor-development.json`: both frozen models, both search methods,
  model/review/search hashes, every detection and aggregate metrics.

The new `mser_chromatic` variant is opt-in research only. Existing evaluator
defaults and all runtime paths stay unchanged. Unit tests cover bounded proposals,
input preservation, dark/neutral frames and invalid evaluation variants.

Next priority is stronger balloon-versus-clutter discrimination and richer
training coverage, not further widening search without classifier support.
The MobileNetV4 teacher has not yet been adapted to the expanded external/proposal
training corpus; evaluating that adaptation before another distillation is a
useful next experiment. Its training path currently validates all splits, whereas
expanded manifests deliberately omit reserved-test pixels: preserve the holdout
boundary when enabling teacher training on those manifests.
