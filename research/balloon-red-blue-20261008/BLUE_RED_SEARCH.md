# Blue-minus-red search: do not promote

2026-10-08. Research-only comparison; Pi/runtime defaults remain unchanged.

The `mser_blue_red` variant changes only the blue MSER plane from positive
`B - max(R, G)` to positive `B - R`. This preserves blue/cyan contrast when the
green channel approaches blue. The red plane, geometry filters, ranking,
deduplication and twelve-candidate budget remain unchanged. Neutral pixels
remain zero, but cyan clutter can also gain contrast. The evaluator applies
the same confirmed-parent suppression as ordinary MSER.

## Training-only proposal check

Same-color IoU >=0.5 coverage, without a classifier:

- Original public training: red stays 19/22; blue improves from 7/15 to 8/15.
- Admitted indoor training: red stays 24/24; blue improves from 20/24 to 22/24.

The original-panel one-pass desktop search time increases from 9.67 to 13.02 ms.
This is neither isolated sustained benchmarking nor a Pi speed estimate.
Training images have incomplete annotations: uncovered regions are not inferred
to be negatives. IMG-family development images are not mined or trained on.

## Full-scene development result

Twelve reviewed indoor scenes, threshold 0.8, IoU 0.5, same frozen models:

- Expanded teacher, ordinary -> blue-minus-red: red TP/FP/FN 11/1/0 -> 11/3/0;
  blue 10/1/2 -> 10/3/2.
- Seed-42 distilled student: red 11/1/0 -> 11/1/0; blue 8/2/4 -> 8/3/4.

No additional true targets were detected. Extra false positives outweigh the
training coverage gains, so the variant is not promoted. Search color groups
generate candidate regions; the classifier may assign either balloon class.
Thus a blue-plane change can also affect red false positives.

These are repeatedly inspected, seen-environment development scenes, not final
qualification. No threshold changed, no final test was evaluated, and no model
was retrained specifically for this search comparison.

## Evidence

- `blue-red-search-original-training.json`: original training coverage/timing.
- `blue-red-search-indoor-training.json`: admitted indoor coverage.
- `blue-red-search-development.json`: full detections, model/source hashes.

Regression test covers cyan recovery, neutral rejection, candidate bound and
unchanged input pixels. Existing default search and deployment code are intact.
