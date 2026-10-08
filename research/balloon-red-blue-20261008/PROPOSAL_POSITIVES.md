# Positive crop coverage: paired refinement

2026-10-08. Research only. The 94% per-color full-scene target remains unmet.

## Training change

Built 30 distinct proposal crops from existing training photos: 19 red and 11
blue. Each uniquely matches a reviewed same-color annotation with IoU >=0.7.
Baseline and MSER searches run at 320x240, with their actual context padding.
Identical crops within a source are deduplicated. No validation or test source
is eligible. Reviewed all 30 crops visually and recorded the hash-bound admission
in `configs/balloon-red-blue-proposal-positive-review-20261008.json`.

These are additional views, not independent new photos or synthetic images.
Truncated, printed, oblique, specular and multiballoon-context examples remain.
Labels inherit the original reviewed annotation, not the model's predictions.

Extended `scripts/refine_pi_student.py` with opt-in positive queues. Original
cached teacher targets remain unchanged. New positive proposals use hard labels
only. A same-class positive control resamples original positive crops using the
same hard-label loss and same number of entries; this avoids crediting the new
crop policy for merely adding training updates or changing the loss mix.

Both runs start from the original 7,763-parameter separable student, seed 42,
Adam 1e-4, eight epochs and 27 batches/epoch. Each has 420 entries:
212 original crops, 88 reviewed-negative repeats and 120 positive repeats.
Control replaces the 120 proposal repeats with same-class original positives.
Both select epoch eight using the unchanged development hard-loss criterion.

Runs:

- `runs/balloon-red-blue-proposal-control-20261008/`
- `runs/balloon-red-blue-proposal-refined-20261008/`

## Result: no measured advantage over the matched control

All four exported candidates (control/intervention, FP32/INT8) have the following
counts on the same four development photos, at threshold 0.8 and same-class
one-to-one IoU >=0.5 matching:

| Scope | Red TP / FP / FN | Blue TP / FP / FN |
| --- | --- | --- |
| Reviewed development crops | 2 / 0 / 0 | 3 / 0 / 0 |
| Full frame, baseline search | 2 / 0 / 0 | 1 / 0 / 2 |
| Full frame, confirmed-parent MSER | 2 / 2 / 0 | 1 / 0 / 2 |

The extra positive training restores the blue crop confidence lost by the
negative-only refinement, and baseline red false detections fall to zero. But
the matched control does equally well: no evidence here that proposal-compatible
views outperform original positive resampling. Blue full-scene recall remains
1/3, and red MSER precision remains 1/2. These tiny reused development counts are
not a qualification result even where they are perfect.

Both candidates accept 2/11 reviewed training negatives, versus 4/11 for the
previous negative-only refinement. This is training fit, not generalization.
Both FP32 exports remain 35,204 bytes; architecture is unchanged. No new Pi
speed measurement was made, and neither new candidate replaced the deployed
research model. The reserved test was not evaluated.

Evidence: `proposal-positive-development.json` contains model hashes, per-frame
proposals, scores, detections and suppression records for all four exports.
Raw crops and run weights remain ignored local artifacts.

## Next evidence needed

More independent real training scenes, especially blue balloons under varied
backgrounds/lighting and non-balloon blue clutter. Repeated tuning against four
photos cannot establish the requested 94% gate. Keep actual Pi camera evidence
separate; the last captured scene was obstructed, not a balloon trial.

Reproduce with `PYTHONPATH=.:src .venv/bin/python`:

1. `scripts/build_red_blue_proposal_positives.py` with the frozen bootstrap
   manifest and a fresh output directory; visually review and hash-bind admission.
2. `scripts/refine_pi_student.py` with the original separable base, the existing
   MSER negative queue/review, new positive queue/review, `--epochs 8
   --negative-repetitions 8 --positive-repetitions 4`; add `--positive-control`
   only for the control run. Never overwrite existing outputs.
3. Export FP32 with `scripts/export_pi_float.py`; evaluate FP32 and INT8 separately
   with `scripts/evaluate_red_blue_development.py` and the unchanged scene review.
