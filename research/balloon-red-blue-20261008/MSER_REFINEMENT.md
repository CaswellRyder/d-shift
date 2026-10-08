# MSER hard-negative refinement: retain as research, not deployment

2026-10-08. The 94% per-color precision and recall target remains unmet. These
results use the same four development photos (two red and three blue targets),
not an independent field benchmark. Reserved-test photos were not evaluated.

## Data admission

Extended the training-only negative miner with opt-in MSER search and a bounded
per-source limit. Baseline behavior remains the default. The separable FP32
student found 19 accepted crops outside upstream balloon boxes. Inspected all
19 thumbnails and all 11 context photos, not just source annotations:

- Admitted 11 crops: clothing, wig, heads, building facade, car lamp, chair fabric.
- Excluded eight crops: six real balloon/balloon-part regions omitted from the
  upstream annotations, plus two ambiguous party-scene regions.
- The blue twisting-balloon sculpture is NOT a safe negative merely because it
  is a different shape. Missing annotations are not proof of background.

The hash-bound decisions are in
`configs/balloon-red-blue-mser-review-20261008.json`. This is AI visual review,
not human certification. Queue/images remain ignored under
`data/balloon-red-blue-mser-review-20261008/`. Source photos are restricted to the
original training partition. Duplicate views do not create independent examples.

## Matched training and actual exported models

Both runs start from `runs/balloon-red-blue-separable-20261008/student.keras`:
7,763 parameters, seed 42, eight epochs, Adam 1e-4, 300 entries and 19 batches per
epoch. The control adds 88 sampled original entries; intervention adds 11 reviewed
negatives repeated eight times. Original teacher targets stay unchanged. New
negatives use hard-label loss only. Each run selects minimum validation hard loss
using the existing eight-crop development set. No runtime thresholds were tuned.

Local runs:

- `runs/balloon-red-blue-separable-mser-control-20261008/`
- `runs/balloon-red-blue-separable-mser-refined-20261008/`

FP32 exports remain 35,204 bytes with unchanged architecture. INT8 was also
exported and checked separately. Neither new candidate was installed on the Pi.

## Full-scene results at the unchanged 0.8 threshold

Same-color one-to-one IoU >=0.5 matching, including search misses and duplicates.
Each cell is TP / FP / FN, not percentages inferred from crop classification.

| Export / training | Baseline red | Baseline blue | Confirmed-parent MSER red | Confirmed-parent MSER blue |
| --- | --- | --- | --- | --- |
| Original FP32 | 2 / 3 / 0 | 1 / 0 / 2 | 2 / 2 / 0 | 2 / 0 / 1 |
| Control FP32 | 2 / 6 / 0 | 2 / 0 / 1 | 2 / 2 / 0 | 3 / 0 / 0 |
| Refined FP32 | 2 / 1 / 0 | 1 / 0 / 2 | 2 / 2 / 0 | 1 / 0 / 2 |
| Original INT8 | 2 / 3 / 0 | 1 / 0 / 2 | 2 / 2 / 0 | 3 / 0 / 0 |
| Control INT8 | 2 / 5 / 0 | 2 / 0 / 1 | 2 / 2 / 0 | 3 / 0 / 0 |
| Refined INT8 | 2 / 1 / 0 | 1 / 0 / 2 | 2 / 2 / 0 | 1 / 0 / 2 |

Reviewed-negative training-fit acceptances fell from 11/11 to 4/11; the control
stayed 11/11. However, refined thresholded blue crop recall fell from 3/3 to 1/3,
even though unthresholded crop argmax accuracy remained 8/8. This is a concrete
reason not to use argmax accuracy as the qualification gate.

The previous confirmed-parent report evaluated INT8. Do not assign its blue 3/3
result to the faster FP32 export: the original FP32 export detects only 2/3 with
that search. Full-scene export behavior must be evaluated independently of both
quantization size and eight-crop host/Pi numerical parity.

Decision: reject promotion of the negative-only refinement. It reduces baseline
red clutter but does not solve blue recall or MSER red false detections. No runtime
default, model pointer, release manifest, or threshold changed.

Evidence: `mser-refinement-development.json` (FP32) and
`mser-refinement-int8-development.json` include model hashes, full proposals,
scores, suppression results and reviewed-negative predictions.

## Fresh Pi check, not qualification

The original separable FP32 model ran alone on the original ARMv6 Pi Zero W for
20 live frames with temporal budget four, requested camera 10 FPS, 180-degree
rotation, 320x240 processing and a selected 640x480 sensor mode. The optimized
TFLite 2.20.0 library was selected per process, not installed system-wide.

- Observed loop: 3.76 FPS over 5.32 seconds; too short for sustained-speed claims.
- Output age p95: 399.75 ms; no invalid/missing ages, increasing timestamps.
- Golden-crop max score delta: 1.19e-7; no thermal/throttling flags.
- Inspected first and last frame: still dark, obstructed/close-up, no identifiable
  balloon scene. This does not establish recall, useful operating range, or safety.
- No ESP32 or actuator commands; capture exited normally.

Raw local evidence under `artifacts/red-blue-pi-results-20261008/`:

| File | SHA-256 |
| --- | --- |
| `live-readiness-view-20261008.json` | `0c6d19c61a9976564c5465380bce3fa5cea9f96030830fbfb52fa7ead6557a34` |
| `live-readiness-view-20261008.frames.jsonl` | `d97c6d5a203047553321b79452a086ae879faec2f5c00cf7767544ef9cd51f45` |

## Next action

Add source-disjoint training coverage, especially positive crops shaped like
actual search proposals, alongside reviewed clutter. Preserve positive confidence
while teaching rejection; do not lower the threshold just to repair these four
reused photos. Keep the reserved holdout untouched until model/policy selection
is frozen. Representative visible Pi balloon scenes remain necessary for the
separate live-perception gate.

Reproduction uses `PYTHONPATH=.:src .venv/bin/python` with:

1. `scripts/mine_red_blue_negatives.py`: bootstrap manifest, original separable
   FP32 model, `--variant mser --per-source 6`, fresh queue directory.
2. Visual review and explicit hash-bound admission (never automatically label
   crops as background from absent boxes).
3. `scripts/refine_pi_student.py`: original separable base, same manifest, queue
   and review; `--epochs 8`; control `--negative-repetitions 0
   --control-repetitions 8`, intervention `--negative-repetitions 8`.
4. `scripts/export_pi_float.py` for each completed run.
5. `scripts/evaluate_red_blue_development.py` for all three models, once per
   export type, using unchanged scene-review and the new negative review.
