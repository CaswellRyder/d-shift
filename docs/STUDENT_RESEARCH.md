# Open architecture research: first measured batch

2026-10-05. Development research only. No camera, motors, Pi package/network
changes, production model replacement, or viewer changes. Five new goal students
were trained and evaluated; the balloon model is unchanged. This is not a claim
of optimality, competition accuracy, or a proven flight system.

## Result

A slightly larger **context + spatial-layout student** is the strongest aggregate
candidate in this batch. It is a custom CNN distilled from MobileNetV4-Conv-Small,
not a full MobileNetV4 running on the Pi. Keep the prior integration candidate
until actual-device latency and new recordings establish the tradeoff.

Same 595 reused development frames, 1,188 labeled goals, 320x240 search and crops,
12 proposals, score threshold 0.8, same-class IoU >=0.5 matching:

| Candidate | Parameters | Crop accuracy, 1,773 crops | Six-class frame precision | Recall | Micro F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Existing 25-epoch integration candidate | 6,263 | 93.97% | 53.33% | 49.92% | 51.57% |
| New controlled tiny, distilled | 6,263 | 86.07% | 52.09% | 38.80% | 44.48% |
| New controlled tiny, supervised | 6,263 | 90.86% | 71.09% | 32.49% | 44.60% |
| Spatial-layout only, distilled | 6,935 | 95.71% | 58.42% | 56.06% | 57.22% |
| Context + layout, supervised | 16,183 | 96.90% | 67.29% | 54.88% | 60.45% |
| Context + layout, distilled | 16,183 | 98.31% | 69.13% | 58.25% | 63.23% |

**Do not conflate crop accuracy with full-frame accuracy.** The deployment problem
includes finding the target, localizing its box, rejecting distractors, identifying
shape/color, tracking, and meeting the camera-to-control deadline.

New controlled runs explicitly reset the random seed after teacher loading/cache
setup. The earlier integration run did not do this, so it is a retained incumbent,
not an exact rerun with identical initialization. The tiny rerun's regression
shows sensitivity to initialization/training trajectory. One seed and reused
correlated development scenes are insufficient for statistical/generalization
claims. No reserved test images were opened by these experiments.

## What changed in the model

Input remains 64x64 RGB with embedded 1/255 scaling. The original stride-two
8/16/32-channel convolution stem is unchanged.

- `tiny`: global-average pool -> seven class logits.
- `spatial`: 2x2 average-pooling layout -> flatten -> seven logits.
- `context`: add a 32-channel 3x3 convolution at the 8x8 feature map, then the
  2x2 layout head. Last convolution receptive field increases from 15 to 31 input
  pixels, before pooling. Retained layout can distinguish arrangements of edges.

These are hypotheses supported by this development ablation, not proof that a
particular feature was learned. This is ordinary architecture engineering, not
a claim of a novel research architecture or a reproduction of spatial pyramid pooling.

The context candidate has 1,401,728 convolution/dense multiply-accumulates versus
811,232 in the original (1.73x). This count excludes pooling, activations and data
movement; **it is not a latency ratio**. FP32 export is 68,064 bytes; INT8 is 22,512.
Both have 98.31% crop validation accuracy. INT8 full-frame micro F1 is 63.42%,
versus FP32 63.23%; this tiny difference does not establish superiority.

## Distillation experiment

Four primary runs form a 2x2 architecture/loss comparison: tiny/context and
distilled/supervised. Same reviewed train/val manifest, class order, seed 42,
25 epochs, batch 32, Adam 1e-3, no augmentation, and checkpoint selection by
validation **hard-label loss**. Distillation uses temperature 4, equal weights on
hard-label CE and temperature-scaled teacher KL. Supervised uses hard labels only.

Teacher logits are cached for identical train crops. Cache reuse verifies teacher
and manifest hashes, class order, finite tensors, target dimensions and hard-label
alignment. Supervised placeholder zeros cannot masquerade as teacher logits.
Class scores are not calibrated probabilities; thresholds were held at 0.8 rather
than retuned independently to make each candidate look better.

Distillation helped the context architecture in this batch. It did not universally
help every metric of the tiny architecture. The supervised context candidate has
better orange-circle precision; aggregate ranking is not per-class dominance.

## The next bottleneck is localization

An oracle diagnostic supplies the annotated target box with 12% context, bypassing
candidate search. These are **positive-only conditional classification results**,
not detection precision/recall or measured-distance performance:

| Model | Correct ID from 320x240 source | Correct ID from 640x480 source |
| --- | ---: | ---: |
| Existing student | 83.67% | 90.15% |
| Offboard teacher | 83.33% | 93.01% |
| New context student | 97.14% | 97.64% |

At 320x240, only 735/1,188 targets (61.87%) have IoU >=0.5 with a top-12
proposal. Allowing 64 diagnostic proposals covers 860/1,188 (72.39%). These are
per-target coverage counts, not a one-to-one detector score or a guaranteed
attainable ceiling. Oracle crop shape/context differs from generated proposals.

For the 284 targets whose longest side is under eight pixels, the context model
correctly identifies 273 when given their boxes; only 15 have top-12 coverage.
This suggests substantial localization headroom on this dataset. It does **not**
establish operational identification range: source scenes are correlated, boxes
are supplied, and no physical distances were measured.

Simply applying unchanged pixel thresholds/kernels at 640x480 reduces top-12
coverage to 693/1,188. Resolution changes require a designed search strategy.

## Paired-resolution experiment

`dtr.detail_vision.observe_detail` searches a small image and classifies crops from
the larger, same-frame image. Boxes are returned in **source-image pixels**, with
explicit source/search dimensions and resolution-independent normalized centers.
If two images are supplied, the caller must guarantee the same frame and field of
view. This module is opt-in and is not connected to the viewer or camera loop.

The experiment kept precisely the frozen 320x240 proposals and compared low-detail
versus 640x480 crops derived from the original images:

| Model | Low-detail full-frame micro F1 | High-detail micro F1 |
| --- | ---: | ---: |
| Existing student | 51.57% | 50.70% |
| Context student | 63.23% | 63.77% |

More detail is not an automatic win. The context model gains only 0.54 percentage
points, while the old model regresses. This method cannot rescue missing boxes.
It remains experimental until camera bandwidth, alignment and Pi latency are tested.

## Class tradeoffs still matter

Context/distilled FP32, normal 320x240 path:

| Class | Precision | Recall |
| --- | ---: | ---: |
| Orange circle | 43.21% | 28.00% |
| Orange square | 62.67% | 23.04% |
| Orange triangle | 42.66% | 46.27% |
| Yellow circle | 76.32% | 91.19% |
| Yellow square | 94.86% | 94.12% |
| Yellow triangle | 94.17% | 81.51% |

Orange-triangle precision regresses from the previous model despite higher recall.
Orange goals remain unsuitable for a claim of competition readiness. Yellow-only
localization, ignoring shape, improves from 83.77% to 89.71% F1; that easier metric
must not replace the six-class result in a report.

## Research directions still open

Prioritize by expected end-to-end benefit, not model branding:

1. **Learned small-target localizer:** compact fully convolutional center/extent
   prediction trained on existing COCO boxes. Compare against the HSV gate, keeping
   the good crop verifier. A tiny class-agnostic localizer is an option; no need to
   immediately replace the whole stack with a large detector. Include sufficient
   context for hollow goal centers. Center-only detections do not provide aperture
   clearance; box/rim geometry still needs estimation.
2. **Calibrated pixels + geometry:** actual Pi-camera color calibration, small-rim
   connected components, tolerant shape fitting and a cheap learned scorer. The
   original yellow RGB demo was not calibrated for these public recordings; its
   poor dataset result does not disqualify the family of pixel methods.
3. **Box refinement / localization distillation:** teach a small box correction or
   localization network using ground truth and the existing offboard detector.
   Teacher detections are not ground truth; evaluate their errors before use.
4. **Search versus tracking schedules:** higher-detail or tiled search on loss,
   cheap tracking between fresh classifications, bounded label/measurement age.
   Budget search latency and wrong-target persistence, not just average FPS.
5. **Generalization work:** color-preserving blur/exposure/scale augmentation,
   hard-negative mining from train recordings, additional seeds, and actual Pi
   camera recordings at multiple distances/angles. Hold out whole new sessions.

Accept a slower candidate if it wins the mission-relevant accuracy/latency tradeoff.
Measure p95 camera-to-result age, acquisition and reacquisition latency, incorrect
class/target persistence, and near-goal behavior. A slow blimp still turns and
approaches obstacles; average FPS alone cannot establish an adequate deadline.
No architecture is locked and this batch does not exhaust possible improvements.

## Research sources

- [Spatial pyramid pooling](https://arxiv.org/abs/1406.4729): motivation for
  retaining spatial structure; our single 2x2 head is a simpler, different design.
- [Teacher-assistant distillation](https://arxiv.org/abs/1902.03393): capacity gaps
  motivate testing the loss/student jointly, not assuming a larger teacher always wins.
- [MobileNetV4](https://arxiv.org/abs/2404.10518): teacher and mobile architecture
  context; published modern-device latency does not establish original Pi Zero speed.
- [FOMO](https://docs.edgeimpulse.com/studio/projects/learning-blocks/blocks/object-detection/fomo):
  constrained-device learned-localization direction, with centroid/grid limitations.
  Not implemented here and no borrowed-device benchmark is a Pi Zero measurement.

## Artifacts, checks and reproduction

All artifacts are under `runs/student-research-20261005/`:

- `plan.json`, `results.json`: completed controlled 2x2 batch; the fifth,
  follow-up `spatial-kd/` run is separate and is included in the table above.
- Per-run `provenance.json`, `epochs.csv`, `report.json`, `.keras`, INT8/FP32 models,
  `frames-mac.json` and per-frame observations retain provenance and errors.
- `oracle-baseline.json`, `oracle-context.json`: conditional crop/coverage diagnostics.
- `detail-first.json`: paired source-detail comparison and all six-class counts.
- `pi-crop-bundle-v2/`: current/context/spatial in both numerical formats, 21 frozen
  reference crops, checksum-bound metadata, runtime adapter and standalone benchmark.
- `crop-bench-mac.json`: six models x 63 invocations, all local parity checks pass.
  **Mac only**; this is a smoke test, not Pi latency or cross-device parity proof.

192 tests passed, two skipped; Ruff passed. Original frozen comparison files are
preserved. No model was promoted or live service restarted. Pi hostname resolution
failed and its former USB address timed out during this turn; new actual-device
latency remains unmeasured. No training job is left running by this batch.

Reproduce the controlled four-run batch into a **new** directory:

```sh
PYTHONPATH=src .venv/bin/python scripts/run_student_research.py \
  --output runs/student-research-repeat --epochs 25
```

After transferring the self-contained crop bundle to an available Pi, run inside it:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 pi_research_crops.py \
  --bundle . --rounds 3 --output crops-pi-01.json
```

The benchmark has no camera or motor code. It refuses to overwrite a prior result,
checks file hashes and numerical parity, alternates model order, uses one inference
thread and records the actual machine architecture. Follow crop timing with matched
full-frame and camera tests before selecting a runtime format or integration model.
