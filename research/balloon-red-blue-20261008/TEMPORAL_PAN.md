# Synthetic-pan reacquisition check for periodic search

2026-10-09. No red/blue balloons are available for live camera tests for some
time, and the room was dark. This experiment studies the scheduler's temporal
behavior offline. It does **not** measure real motion, occlusion, motion blur,
lighting change or flight performance. No weights, thresholds or deployment
settings changed. No ESP32, serial, motor or network-configuration commands
were issued.

## Method

`scripts/research_temporal_pan.py` uses the twelve admitted EngDes2 development
photos. All are from the IMG_ family, which is evaluation-only, taken from the
upstream `valid` split. Excluded frames and the reserved final test were not
used. Each 640x640 export is viewed through a deterministic moving square window
of 320 to 416 source px, rendered with INTER_AREA to 320x240, so it never
upsamples. Window center and size follow fixed sinusoids, with the phase offset
per scene. Targets therefore leave and re-enter view. Each scene runs 12 s at
10 FPS, giving 120 views per scene and 1440 overall.

- **Provenance checks.** A full-frame window reproduces every reviewed 320x240
  frame byte-for-byte. Desktop detections on those frames match the scheduled
  Pi bundle's goldens for both modes and both seeds.
- **Labels.** Reviewed boxes are mapped to source pixels and clipped to the
  window.
  - A target counts as present when at least 75% of it is visible and the
    clipped box is at least 4 px on each side.
  - Partly visible targets are ignore regions: accepted same-color boxes mostly
    inside one count as neither TP nor FP.
  - Matching otherwise follows the development rule: confidence-ordered and
    one-to-one at IoU 0.5, with wrong colors and duplicates counted as false
    positives.
- **Scoring.** Both `mser_direct` (full) and `mser_bright` are scored on every
  view.
- **Policy replay.**
  - The `full` and `periodic` policies are replayed through the unchanged
    `SearchSchedule`.
  - Frame k is exposed at k*100 ms and delivered 120.7 ms later. That delay is
    the median delivery time in the dark live log, whose saturated 66.7 ms
    exposure is included.
  - Each search takes the first frame delivered after the previous result.
    This models `queue=False`.
  - Search costs are constants: the Pi periodic-replay means of 320.3 ms for
    full and 244.6 ms for bright. Content-dependent cost is not modeled.
- **Episodes.**
  - An episode is a maximal run of views in which a target is present. There
    are 72, and the three balloon-free scenes contribute only false positives.
  - First-detection latency runs from the episode's first exposure to the
    result of the first frame that detects that target.
  - Confirmed-observation age is the time, at each later result in the episode,
    since the newest exposure that detected the target.

## Results

| seed | policy | frames/s | episodes processed | episodes detected | first detection mean / p95 / max (ms) | confirmed age p95 / max (ms) | frame FP |
|---|---|---|---|---|---|---|---|
| 42 | full | 2.50 | 66/72 | 56 | 587 / 766 / 841 | 841 / 1641 | 126 |
| 42 | periodic | 2.92 | 72/72 | 61 | 549 / 665 / 865 | 591 / 1465 | 147 |
| 43 | full | 2.50 | 66/72 | 54 | 578 / 741 / 1041 | 441 / 1641 | 148 |
| 43 | periodic | 2.92 | 72/72 | 59 | 555 / 673 / 841 | 741 / 1465 | 172 |

**Per-view mode agreement.** Over all 1440 views, bright-only and full search
had identical true-positive sets for both seeds: zero full-only and zero
bright-only hits. True positives were 764 for seed 42 and 761 for seed 43.
Bright-only produced slightly fewer false positives, 470 vs 500 for seed 42 and
569 vs 584 for seed 43. The known dark-on-bright omission is therefore **not
represented** in these scenes. That shows no cost on this material; it does
not show the cost is zero in general.

**Interpretation.** Here the periodic policy's advantage comes entirely from
processing about 17% more frames.

- It reached every episode, including six short ones the full policy never
  sampled, and detected five more episodes.
- Its first-detection mean and p95 are lower for both seeds.
- Confirmed-age p95 is mixed: better for seed 42 and worse for seed 43. Its
  maximum is lower for both seeds.
- Its false-positive count rises roughly in proportion to frames processed.
  The false-positive rate per frame is about the same.

Persistently missed episodes were the same targets under both policies:
scene 3 target 1 and scene 6 targets 0 and 2. Those are detector recall
failures, not scheduling failures. The extra periodic "missed" entries are
1-frame episodes that the full policy never processed at all. Maximum
confirmed ages of 1.5–1.6 s come from intermittent detector misses while a
target stays in view. Neither policy guarantees a reacquisition deadline.

## Limitations

- All scenes are still photos from a seen environment, so the pans are not real
  motion. There is no blur, rolling shutter, exposure change or balloon motion.
- Costs are constants and delivery latency comes from the dark scene. Lit-scene
  delivery is expected to be shorter.
- Twelve scenes from one development family are not independent qualification
  data.
- Occlusion was not simulated.

Evidence: `temporal-pan-development.json`, which holds the timing-model
sources, trajectory, per-scene episodes, per-policy schedules and summaries.
Unit tests are in `tests/test_research_temporal_pan.py` (7). Reproduce:

```sh
.venv/bin/python -m scripts.research_temporal_pan --base artifacts/balloon-pi-scheduled-search-20261008 --library artifacts/balloon-regions-mac-20261008/regions.so --archive data/raw/engdes2-red-blue-v1-20261008/dataset.zip --review-root data/engdes2-development-scenes-20261008 --decision configs/engdes2-development-scenes-20261008.json --family configs/engdes2-development-family.json --pi-replay artifacts/balloon-pi-scheduled-results-20261008 --live-log artifacts/balloon-live-scheduled-results-20261009/full-a/frames.jsonl --model new-views-42 --model new-views-43 --output artifacts/fresh-temporal-pan.json
```

Still pending: the lit live-camera comparison (`PERIODIC_SEARCH.md`), and real
target sequences for appearance, occlusion and loss once balloons are
available.
