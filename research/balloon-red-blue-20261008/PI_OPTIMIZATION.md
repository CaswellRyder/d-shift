# Red/blue Pi optimization — 2026-10-08

Status: research candidates, not deployment or flight approval. The target remains
at least 94% precision **and** recall per color on representative held-out scenes.
The current four-photo development set cannot establish that result.

## Factorization experiment

This is **not Samsung LittleBit**. It reuses our existing convolution-specific
rank-one SVD initializer: replace the context 3×3 convolution with depthwise and
pointwise convolutions, copy compatible stem/classifier weights, then distill
again. No new binary runtime or Samsung code was introduced.

The original red/blue context model has 15,667 parameters; the separable model
has 7,763. FP32 artifacts are 66,000 vs 35,204 bytes; INT8 artifacts are 21,888 vs
15,656 bytes. The warm-start kernel approximation has 0.8145 relative error, so
it is not an equivalent graph rewrite. The candidate was retrained for 25 epochs
using unchanged training-only teacher logits, temperature 4, hard-label weight
0.5, seed 42, and the existing development checkpoint-selection rule.

```sh
.venv/bin/python scripts/distill_pi_student.py \
  --config configs/balloon-red-blue.json \
  --manifest data/balloon-red-blue-bootstrap-20261008/manifest.json \
  --teacher runs/balloon-red-blue-teacher-20261008/teacher.keras \
  --output runs/balloon-red-blue-separable-20261008 \
  --epochs 25 --student-variant separable_context \
  --target-cache-run runs/balloon-red-blue-context-20261008 \
  --initialize-context runs/balloon-red-blue-context-20261008
.venv/bin/python scripts/export_pi_float.py \
  --run runs/balloon-red-blue-separable-20261008 \
  --output runs/balloon-red-blue-separable-20261008/student.float32.tflite
```

`separable-training.json` binds the training lineage and INT8 export.
`separable-development.json` records all proposal variants and detections.
The separable model retains 8/8 crop argmax accuracy, but the baseline full-frame
profile still has red TP/FP/FN = 2/3/0 and blue = 1/0/2. MSER recovers three blue
targets but introduces nine red false positives. Neither result qualifies.
The refined context candidate has a separate hard-negative training intervention;
comparing it with unrefined separable is not a pure architecture ablation.

## Pi isolation and reproducibility

- Actual original Pi Zero W, `armv6l`, OV5647 camera detected.
- Read-only process check found no competing Python/rpicam vision workload before
  tests. Existing desktop/system services remain running; no service tuning.
- One model/policy per Python process, sequential shell loop; no simultaneous
  competing detection implementations. Startup and model loading excluded from
  crop/replay timing, and reported separately from live end-to-end loop speed.
- One inference thread, OpenCV one thread; unchanged confidence threshold 0.8.
- The existing rebuilt TFLite 2.20.0 library is selected per process with
  `DTR_TFLITE_LIBRARY`; the installed system library is not replaced.
- Rebuilt library SHA256:
  `54ed7fdda0b5b5f6db05a538eb800b7bc7ecd70d32f67dc13d1e78817d8b653a`.
- All eight golden development crops must match the desktop label/acceptance
  decision and score tolerance before a timing run begins.
- Crops: 80 warmed timed predictions, cycling eight crops. Replay: 20 timed
  frames, cycling four photos after a full warm-up pass. Repetitions are timing
  samples, NOT additional independent accuracy examples.
- Camera: 320×240 RGB processing, requested 10 FPS, two buffers, 180° transform,
  bounded four-inference temporal policy. Raw first/last frames stay in ignored
  artifacts, not the repository. No ESP32, serial, or motor commands.
- Live frame age uses Linux CLOCK_BOOTTIME, matching libcamera SensorTimestamp:
  [libcamera control definition](https://docs.libcamera.org/master/internal-api/namespacelibcamera_1_1controls.html).

The new bundle builder includes only development inputs, never reserved test
images. v2 was used for crop timing; v3 corrects the live timestamp clock before
any camera run. Models and input hashes are identical across both bundles.
Bundle manifests and `pi-inputs.json` are retained here. Existing Pi installations
and all earlier evidence remain unchanged.

Run the bundle's `pi_red_blue_bench.py` with `--base`, `--model` (alias in
`inputs.json`), `--mode crops|replay|camera`, a fresh `--output`, and bounded
`--count`. Camera temporal policy additionally uses `--policy temporal --budget 4`.
Copy completed result JSON and JSONL files back, then summarize with:

```sh
.venv/bin/python -m scripts.summarize_red_blue_pi \
  --results artifacts/red-blue-pi-results-20261008 \
  --inputs artifacts/red-blue-pi-20261008-v3/inputs.json \
  --output research/balloon-red-blue-20261008/pi-measurements.json
```

## Measured results

All values below were measured on the actual ARMv6 Pi, not projected from Mac
timing. `pi-measurements.json` retains reports and binds raw logs by checksum.
Raw JSONL and camera snapshots remain in `artifacts/red-blue-pi-results-20261008`.

| Model | FP32 crop mean | INT8 crop mean | FP32 full-frame replay |
| --- | ---: | ---: | ---: |
| Context | 29.44 ms | 57.50 ms | 320.92 ms / 3.12 processing FPS |
| Refined context | 29.75 ms | 57.51 ms | 325.86 ms / 3.07 processing FPS |
| Separable | 24.27 ms | 44.83 ms | 269.33 ms / 3.71 processing FPS |

Separable reduces measured FP32 crop latency by 17.6% and increases full-frame
replay throughput by 19.2% against the original context model. This is one crop
and replay run per configuration, not a statistical confidence interval. INT8
remains slower than FP32. Per-crop model-file size is not the bottleneck.

The baseline search/profile produces the same tiny-set full-frame counts on
the Pi as on the desktop: context and separable red TP/FP/FN 2/3/0, blue 1/0/2;
refined context red 2/1/0, blue 1/0/2. Better crop scores have not solved search
recall or red clutter rejection. No candidate is promoted.

| Live temporal model | Run A loop FPS | Run B loop FPS | A / B p95 output age |
| --- | ---: | ---: | ---: |
| Context FP32 | 4.17 | 4.32 | 485.9 / 476.9 ms |
| Separable FP32 | 5.00 | 5.25 | 433.7 / 406.9 ms |

Live order was context A, separable A, separable B, context B; 120 frames per
run, single process at a time. Both models used the same four-inference temporal
budget, 180-degree camera transform, and requested 10 FPS. The sensor selected
640×480 with 320×240 processed output; **these are not full-resolution 2592×1944
processing benchmarks**. About 123 MiB peak process RSS was recorded on Linux.
All four runs had strictly increasing sensor timestamps and no missing/negative
frame-age values. No throttling/undervoltage flags were reported. The existing
cpuinfo topology warning appeared on INT8 load but did not prevent parity checks.

**Camera limitation:** viewed first/last samples show a dark, mostly obstructed
close-up, not an arena or clearly visible balloons. Zero accepted observations
in the first two live runs are not evidence of useful target recall. Live FPS
and age depend on scene/search workload; these short tests are not a sustained
flight trial or a guarantee that all frames meet a deadline. Saved samples are
not submitted as a labeled accuracy set.

Replay stage means explain the remaining speed ceiling: context search 100.45 ms,
crop/classification 219.26 ms, assembly/suppression 1.22 ms. Separable keeps search
at 99.55 ms and reduces crop/classification to 168.65 ms. Next, profile mask and
contour construction and test equivalence-preserving acceleration before changing
color thresholds or sacrificing small-target coverage. Further weight compression
cannot remove that approximately 100 ms search cost.

No default model/profile promotion is made by this experiment. Real independent
Pi balloon recordings, clutter/negative scenes, range calibration and the team’s
latency/flight-acceptance requirements remain outstanding.
