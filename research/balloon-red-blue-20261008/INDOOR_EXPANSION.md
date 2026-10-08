# First reviewed external-source training expansion

2026-10-08. Research candidate only. No deployment promotion; 94% per-color
precision and recall on representative held-out scenes is not established.

## Admission and leakage boundaries

Selected 16 frames, eight from each of two broad filename families in the
EngDes2 export: `frame_####` and `frame_######`. The first family shows balloons
against patterned walls; the second includes kitchens, seating, garages and
outdoor pavement. The script's `indoor` name does not mean all frames are indoor.
All selected families are training-only, not independent validation sessions.

Selected one export variant per basename, excluded basenames that also appear in
the upstream valid/test metadata, and screened exact full-frame pixel duplicates
within the selection. Automated 63-bit grayscale DCT perceptual hashing compared
each selected image under all four horizontal/vertical flip combinations with
the original corpus's 17 validation/test source photos. Nearest Hamming distances
were 18..24 (reject threshold <=8); no candidate was flagged. This heuristic does
not prove unrelated capture sessions or absence of every possible transformation.

**Holdout pixel handling:** the automated duplicate screen read original holdout
source pixels only to compute fingerprints. No holdout previews, model predictions,
labels for tuning, or training examples were produced. The new corpus's upstream
valid/test pixels were not read. This is distinct from claiming no holdout file
was ever opened. The reserved model test remains unevaluated.

Visually inspected all 16 full frames and 80 cropped proposals. Excluded crop 3
(a genuine partial blue balloon mislabeled background by missing annotations) and
crop 78 (ambiguous blurred colored region). Admitted 78 crops: 24 red, 24 blue,
30 non-balloon background. This is AI review, not human certification. Labels
apply only to these crops, not complete-frame object counts.

Queue: `data/engdes2-indoor-review-20261008/review.json`, SHA-256
`4049fcac260440de93e061167706574098804f184072da22b60fd07053fab193`.
Decision: `configs/engdes2-indoor-review-20261008.json`.
The generic upstream `balloon` class and stock-photo filename families were not
admitted. Original license/source limitations remain in PUBLIC_DATA_EXPANSION.md.

## Matched training

Each dataset contains the original 212 training entries plus 156 additions:

- Intervention repeats each new reviewed crop twice.
- Control draws 156 original training crops, matching the intervention's added
  class counts and optimizer update count.
- Both have 368 training entries: 198 background, 92 red, 78 blue; original
  validation labels and source photos remain unchanged.
- Reserved test metadata is retained for leakage checks, but its crop files are
  not copied into either new dataset.

`scripts/distill_pi_student.py` now supports an explicit same-architecture student
warm start and bounded learning rate. Both experiments start from the same frozen
7,763-parameter separable checkpoint, use Adam 1e-4, seed 42, eight epochs, 23
batches/epoch and hard-label loss (`alpha=1`). This is supervised refinement of
an already-distilled student, **not new teacher distillation**. Teacher logits are
unused zero placeholders and are marked as such, not fabricated teacher evidence.
Validation hard loss selects epoch eight for both runs.

Manifests:

- Control: `data/balloon-red-blue-indoor-control-20261008/manifest.json`, SHA-256
  `22f69d7d023b4c6e0d184123580142a7f7a1466428119bce3590c0e15a35f420`.
- Expanded: `data/balloon-red-blue-indoor-expanded-20261008/manifest.json`, SHA-256
  `93928e1dbe063b9230433615b7beea1bdd8cc21f640bf82b9793f1d5237c8350`.

Run directories have the same suffixes under `runs/`. Architecture and export
sizes remain unchanged: FP32 35,204 bytes, INT8 15,656 bytes. No new Pi timing
was measured; no model pointer, camera configuration or runtime default changed.

## Fit on the new training crops (not generalization)

| Model, FP32 | Red correctly accepted /24 | Blue correctly accepted /24 | Background falsely accepted /30 |
| --- | ---: | ---: | ---: |
| Original separable | 21 | 14 | 0 |
| Matched control | 23 | 16 | 1 |
| Expanded training | 23 | 22 | 0 |

This demonstrates learning from the additional source material, not held-out
accuracy. All predictions and hashes are in `indoor-training-fit.json`.

## Existing four-photo development check

The development set contains only two red and three blue targets. The following
counts hold for BOTH FP32 and INT8 exports, at unchanged threshold 0.8 and
same-color one-to-one IoU >=0.5 matching:

| Candidate | Baseline red TP/FP/FN | Baseline blue TP/FP/FN | Confirmed-parent MSER red | Confirmed-parent MSER blue |
| --- | --- | --- | --- | --- |
| Control | 2/5/0 | 2/0/1 | 2/2/0 | 3/0/0 |
| Expanded | 2/4/0 | 2/0/1 | 2/2/0 | 3/1/0 |

Original baseline was red 2/3/0, blue 1/0/2. New candidates trade higher blue
recall for more red false detections. The control also reaches MSER blue 3/3,
so the expanded source cannot be credited for that recovery on these photos.
The expanded candidate adds a blue false detection under MSER. It is not a
qualified replacement. Full records: `indoor-expansion-development.json`.

## Next work

Combine the new positive coverage with the previously reviewed red/blue clutter
negatives; the current expansion's 30 background crops do not cover the known
hard-negative failure modes. Expand independently grouped development scenes
before further model selection, preserving a final untouched test. Representative
unobstructed Pi recordings remain required for live qualification. No ESP32 or
motor commands were issued.

Reproduction uses `PYTHONPATH=.:src .venv/bin/python`:

1. `scripts/prepare_indoor_balloon_review.py` with the downloaded public root and
   original bootstrap manifest; review every crop/context and bind admission.
2. `scripts/build_indoor_balloon_training.py` with that queue/review and original
   manifest, once with `--control`, once without; fresh output directories.
3. `scripts/distill_pi_student.py` for each manifest, red-blue config and frozen
   teacher path; `--student-variant separable_context --initialize-student
   runs/balloon-red-blue-separable-20261008 --learning-rate 0.0001 --alpha 1
   --epochs 8`.
4. Export FP32, then run `scripts/evaluate_red_blue_development.py` on all four
   exports using the original scene-review/manifest. Separately run
   `scripts/score_indoor_balloon_fit.py` on the admitted training queue.
