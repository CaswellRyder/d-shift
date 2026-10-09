# Lit camera checks and tiny-proposal context experiment

2026-10-08. The user turned the room lights on but has no goals/balloons available.
This enabled live clutter/timing investigation, **not target recall, range or
flight qualification**. No weights, deployed thresholds or production policies
changed. No ESP32 or motor commands were issued.

## First lit run

Same original Pi Zero W, current seed42 student, optimized MSER, full 2592x1944
sensor mode, 320x240 RGB processing, requested 10 FPS and 180-degree transform as
the preceding dark-scene check. No concurrent vision implementation.

- 40 frames, 12.317 seconds: **3.248 observed live FPS**.
- Mean processing 250.619 ms; search 208.569 ms, inference 41.637 ms.
- Mean sensor-to-result 373.472 ms; p95 390.927 ms; maximum 434.530 ms.
- All 40 sensor timestamps valid and strictly increasing.
- Mean 1.125 neural calls/frame; four accepted blue detections, zero red.
- Throttle flags `0x0`; maximum process RSS 124,756 KiB.

Blue detections occurred at frame indices 5, 10, 14 and 25. Their boxes were
respectively [311,27,319,32], [311,27,319,33], [290,33,302,37] and [312,28,319,32],
with probabilities 0.999995, 0.999995, 0.999493 and 0.999999 (rounded).
They align with the blue display edge in neighboring saved frames, and the user
reported no balloons available. These are apparent clutter false positives,
not examples where a modestly higher confidence threshold is an adequate fix.

Important limitation: that harness version retained only frames 0 and 39, neither
of which contained an accepted detection. **Exact event pixels were not retained**;
do not substitute neighboring images as though they were the detected frames,
or admit these four events as reviewed training crops. No labeled false-positive
rate or independent accuracy estimate is claimed from this short scene.

The saved view is illuminated but tilted, with much of the view occupied by the
display/mount. The hardware transform is verified; physical mounting orientation
and arena field of view remain unqualified.

## Bounded event capture and second run

Updated the research camera harness to retain first/last images plus the first
eight accepted-detection frames. Arrays already owned by the loop are retained
in memory; all PNG/JSON writes still happen after timing. At most ten images are
kept. The reviewer checks bounded indices, hashes and brightness for every saved
sample. Unit tests verify event limits and mandatory retention of the last frame.

The updated script was copied under a new Pi filename,
`pi_balloon_live_bench_events.py`, preserving the earlier capturing script.

The second run completed 60 frames in 12.113 seconds (4.953 FPS) with **zero
proposals/neural calls**. The visible view had changed: the blue display content
from the earlier view was no longer present. Its 284.836 ms mean sensor-to-result
age and higher loop rate describe this easier scene, not a model improvement.
Both saved samples were lit (not nearly black), and all timestamps were valid
and increasing. No event sample was triggered; no new training labels were made.

This is why workload and neural-call counts must accompany live FPS. The earlier
paired 16-scene replay remains the evidence for the optimization's speed gain.

## Minimum-context ablation

Hypothesis: resizing a tiny, almost-uniform blue patch to the model's 64x64 input
can hide its surrounding screen-edge context. Test adding context without
discarding small candidate boxes or raising the classifier threshold.

`scripts/research_balloon_context.py` applies one fixed rule: if **both** dimensions
of the padded candidate crop are smaller than 16 pixels, expand it to a bounded
16x16 window containing the original crop. Candidate boxes, proposal ranking,
search budget, weights, suppression and 0.8 threshold stay unchanged. Larger
crops remain unchanged. This is a single exploratory ablation, not a parameter
sweep or evidence that 16 is optimal. No training was performed.

Both frozen student seeds were evaluated on the existing four original and
twelve indoor development scenes. The original arm was verified against the
previous golden predictions before comparison. Entire IMG-family scenes stayed
evaluation-only; no reserved final test was opened, mined or trained on.

The rule changed 22 candidate crops per model but changed **no aggregate
TP/FP/FN counts**:

- Seed42 indoor: red 11/0/0, blue 9/4/3; original: red 2/2/0, blue 3/0/0.
- Seed43 indoor: red 10/2/1, blue 8/4/4; original: red 2/2/0, blue 3/1/0.

Several rejected candidate labels/scores changed, so this is not prediction
parity. The existing accepted tiny red candidate remained accepted in both
seeds. That one observation is insufficient to establish distant-target recall.
The two saved lit samples have only a large background proposal, so they do not
test the intended screen-edge event correction at all.

**Do not promote this rule:** it has not improved the measured counts or been
tested on the actual live false-positive pixels. No Pi latency result is claimed
for changed cropping. Keep the exact speed optimization, and keep this separate
context experiment opt-in until better evidence exists.

## Current error direction

### Subsequent lit-room recheck

An additional isolated capture used the unchanged `new-views-42` model and
event-retaining harness: 60 frames in 24.700 seconds, **2.429 live FPS**. Mean
processing was 338.122 ms, of which search consumed 337.418 ms. Mean
sensor-to-result age was 484.618 ms (p95 631.736 ms). All 60 timestamps were
valid and strictly increasing; throttle flags remained `0x0`. No competing
Python/camera process was observed immediately before or after the run.

There were zero neural calls and zero accepted detections. Both retained
first/last samples passed the brightness diagnostic; visual inspection of the
first confirms a lit but tilted room view. This is not an independent labeled
negative set or a target-recognition test. The different timing from earlier
captures is not a measured code regression: scene and capture conditions were
not held fixed. Search dominates this run and deserves profiling on fixed
replay inputs before changing its recall-sensitive proposal policy.

Raw artifacts remain ignored under
`artifacts/balloon-live-lit-recheck-20261008`; the hash-verified review is
`fast-search-live-lit-recheck.json`. Camera-loop and artifact-review tests:
23 passed. No weights, deployment settings, training admissions, or actuator
interfaces changed. Password-based SSH succeeded; key-only authentication did
not. No SSH configuration was modified.

Reinspection of the existing seed42 error report confirms that its three indoor
blue misses split into two with no localized proposal and one with a localized
proposal classified incorrectly. Its four blue false positives remain. A global
confidence change cannot create missing proposals or reliably reject the very
confident display-edge errors. Further work needs reviewed tiny-clutter/balloon
examples and proposal coverage, with source-group separation and explicit data
admission. Newly captured images must not silently become both training and
qualification examples.

## Evidence and reproduction

- `fast-search-live-lit-scene.json`: first lit run plus hash/timestamp review.
- `fast-search-live-events-scene.json`: second run and bounded-capture metadata.
- `minimum-context-development.json`: both seeds/arms, complete detections,
  model identities and unlabeled saved-sample diagnostics.
- Raw pixels/logs remain ignored under `artifacts/balloon-live-lit-a-20261008`
  and `artifacts/balloon-live-lit-events-a-20261008`, and in corresponding Pi
  research output directories. No camera images are committed or published.

Reproduce the context experiment with a fresh output:

```sh
.venv/bin/python -m scripts.research_balloon_context --base artifacts/balloon-pi-fast-search-20261008 --live-samples artifacts/balloon-live-lit-a-20261008 --output artifacts/fresh-minimum-context.json
```

Use the bounded camera command in `LIVE_CAMERA_CHECK.md`, the updated event
script, and a new output name for future captures. A new camera capture is not
an exact replay of these scenes. No current result establishes 94% flight readiness.
