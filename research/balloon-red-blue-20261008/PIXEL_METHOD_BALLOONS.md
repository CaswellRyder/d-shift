# Pixel method vs neural pipeline on balloon frames — actual Pi

2026-10-09. This is a replay on development frames, **not** flight qualification.
No camera was used, and no ESP32, serial, motor or network-configuration
commands were issued. No weights, thresholds or deployment settings changed.

## Why

The only earlier comparison with the other team's pixel method,
`docs/COMPETITION_COMPARISON.md`, measured **yellow goals**. The pixel code on
the Pi (`~/Documents/TESTING/demo.py`, sha `f46b1dfd…`) is a single-color
RGB-distance mask followed by run-length connected components. It ships with
only the yellow goal color (156,156,0), threshold 90, and has no balloon
setting. Our balloon notes so far compared only our own pipelines.

## Method

`scripts/pi_pixel_balloon_bench.py` imports their unmodified `demo.py` by path,
uses only its `Detector`, and checks the file's hash before and after.

- **Backend.** NumPy LUT mask. Numba is not installed on the armv6 Pi.
- **Colors.** One detector per color, run sequentially on the same RGB frame.
  Each reference color is the median, over training-split crops of the
  reference student's dataset, of the per-crop median of the central 50%:
  - red (178,28,48), from 205 crops;
  - blue (26,86,179), from 166 crops.

  No IMG_ development images and no reserved-test crops were read. The other
  team never supplied balloon colors, so these are ours.
- **Thresholds.** 90 is their shipped value and is the primary setting. 60 and
  120 are a sensitivity sweep, not a tuned choice: they were scored on the same
  development frames.
- **Minimum blob size.** Their 200 px at 540×540 is scaled to the same image
  fraction: 53 px at 320×240.
- **Frames.** The same 16 frames as our scheduled-search Pi replay (12 indoor
  IMG_ development frames plus 4 original), identical `inputs.json`.
- **Runs.** One warm-up pass, then 5 timed passes; outputs were identical across
  passes. The Pi and Mac produced identical boxes.
- **Scoring.** The development rule from `evaluate_red_blue_development`:
  one-to-one matching at IoU 0.5; wrong colors and duplicates count as false
  positives. Box area stands in for the missing confidence.
- **Model reference.** Model counts come from the bundle goldens. Model FPS
  comes from the existing Pi replay of the same bundle
  (`PERIODIC_SEARCH.md`, full-a/b and periodic-a/b, seed 42).
- **Pi health.** 39.5→44.4 °C, `throttled=0x0`, idle before the run.

## Results — 320×240, both colors

| Method | FPS (processing) | mean / p95 ms | Indoor red | Indoor blue | Original red | Original blue | All TP / FP / FN |
|---|---:|---:|---|---|---|---|---|
| Pixel, threshold 90 (shipped) | 6.54 | 152.8 / 195.0 | 9/20/2 | 11/18/1 | 2/12/0 | 3/12/0 | 25 / 62 / 3 |
| Pixel, threshold 60 (sweep) | 6.98 | 143.2 / 179.7 | 10/5/1 | 9/11/3 | 2/4/0 | 3/4/0 | 24 / 24 / 4 |
| Pixel, threshold 120 (sweep) | 3.13 | 319.6 / 643.0 | 6/89/5 | 11/81/1 | 1/47/1 | 2/44/1 | 20 / 261 / 8 |
| new-views-42, full MSER search | 3.20 | ~312 / ~544 | 11/0/0 | 9/4/3 | 2/2/0 | 3/0/0 | 25 / 6 / 3 |
| new-views-42, periodic search | 3.60 | ~278 / ~523 | same detections per mode | | | | |
| new-views-43, full MSER search | — | — | 10/2/1 | 8/4/4 | 2/2/0 | 3/1/0 | 23 / 9 / 5 |

The 28 targets are indoor 11 red and 12 blue, plus original 2 red and 3 blue.
At the shipped threshold, the pixel method finds as many targets as seed 42
(25 of 28), with about 10 times as many false positives (62 vs 6). Precision is
29% vs 81%.

- **Per color.** The red pass takes 73 ms and the blue pass 78 ms. Cost grows
  with mask density, which is why threshold 120 is slower.
- **At its native 640×480** (upsampled frames, timing only): 498 ms mean,
  613 ms p95, so 2.01 FPS for both colors. That is slower than our 320×240
  pipeline.

## Interpretation

At equal 320×240 input, the pixel method processes about twice as fast as our
full pipeline (6.5 vs 3.2 FPS) and about 1.8 times as fast as periodic search.
At its shipped threshold it is not competitive on false positives. It has no
classifier, so every red or blue patch of 53 px or more becomes a box.

The threshold-60 sweep cuts false positives to 24 with similar recall. That is
still four times our false-positive count, and the choice was made with
development labels in view. Speed alone favors pixels; precision clearly
favors the neural pipeline.

A pixel mask could serve as our proposal source, but this run does not measure
that. Threshold 60 averages 3.0 boxes per frame at 143 ms, versus MSER's
~203 ms search producing 5.6 classifier calls. Whether the classifier would
remove the pixel false positives without losing recall is untested.

## Limitations

- Sixteen reused development frames, one environment, AI-reviewed labels; not
  independent data.
- The colors and minimum-size scaling are our adaptation. The other team may
  tune balloons differently.
- No temporal ROI memory. `demo.py` has none, and `hollow_test_again.py` is
  goal-shape specific.
- Processing time only. It excludes capture, conversion and resizing, so it is
  not live FPS.

Evidence: `pixel-method-balloons-pi.json` (raw Pi report: per-frame boxes and
timings, health) and `pixel-method-balloons.json` (scored review). Tests:
`tests/test_pixel_balloon_bench.py` (3). Reproduce:

```sh
# On the Pi (single process; nothing else running):
python3 -u pi_pixel_balloon_bench.py --pixel ~/Documents/TESTING/demo.py --pixel-sha256 f46b1dfde183e4695a9daa00ff8d3e9ca01a5ab1c85e969b9992b332a58ce8a7 --base ~/balloon-pi-scheduled-search-20261008 --red 178,28,48 --blue 26,86,179 --threshold 60 --threshold 90 --threshold 120 --native-threshold 90 --repeats 5 --output pixel-balloon-pi.json
# Locally:
.venv/bin/python -m scripts.review_pixel_balloon_bench --base artifacts/balloon-pi-scheduled-search-20261008 --pixel-report pixel-balloon-pi.json --pi-replay artifacts/balloon-pi-scheduled-results-20261008 --output review.json
```
