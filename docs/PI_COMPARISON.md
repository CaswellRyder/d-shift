# Two-method Pi comparison — 2026-10-05

Research bench candidates, not flight approval. The original Pi implementation is preserved
at `/home/pacman/Documents/TESTING/demo.py`; no edits or package replacements are required.
Comparison files live separately at `/home/pacman/pi-comparison-20261005`.

This report preserves the original eight-epoch INT8 baseline. See
[Pi optimization experiments](PI_OPTIMIZATION.md) for the separately trained
25-epoch goal candidate, FP32 runtime measurements, and temporal tracking trials.

## What is being compared

| Method | Implementation | Capabilities in this configuration |
| --- | --- | --- |
| Existing baseline | RGB Euclidean-distance LUT, NumPy run-length connected components | Boxes near RGB (156,156,0), threshold 90, minimum 200 pixels; no shape classification |
| Alternative | OpenCV proposals, 64x64 RGB tiny CNN distilled from MobileNetV4-Conv-Small, full INT8 | Goal color/shape or balloon color classification; rejects predicted background; at most 12 crops/frame |

The Pi student itself is **not** a full MobileNetV4 backbone. It contains three strided
convolutions (8/16/32 channels), global pooling and a logits layer. Training uses real
reviewed proposal-adaptation crops, cached frozen-teacher logits, T=4, equal hard-label
and temperature-scaled KL weights, Adam 0.001, eight epochs, no augmentation. Selection
uses validation hard-label cross-entropy. Calibration uses 500 training crops only.
Teacher training and all distillation occurred offboard. Reserved test pixels were not read.

| Student | Parameters | INT8 bytes | FP32 crop accuracy | INT8 crop accuracy |
| --- | ---: | ---: | ---: | ---: |
| Balloon | 6,131 | 10,640 | 99.41% | 99.41% |
| Goal | 6,263 | 10,880 | 80.60% | 80.49% |

Crop accuracy includes background and does **not** measure full-frame detection. In particular,
the first full-frame student run has poor goal results and many balloon false positives.
Do not carry the earlier YOLO11n precision/recall table into this model's results.

Full-frame **same-class** development results (595 frames, IoU >=0.5, stored threshold 0.8):

| Class | Precision | Recall |
| --- | ---: | ---: |
| Green balloon | 58.66% | 79.48% |
| Purple balloon | 22.25% | 96.26% |
| Orange circle | 17.37% | 13.20% |
| Orange square | 29.87% | 11.27% |
| Orange triangle | 96.30% | 12.94% |
| Yellow circle | 0.00% | 0.00% |
| Yellow square | 98.92% | 72.16% |
| Yellow triangle | 62.50% | 84.03% |

These are the Mac TFLite frame-evaluation results, not new Pi-camera accuracy. They show
that passing the pipeline and golden-output checks does not establish useful competition
accuracy. A confident wrong-shape yellow prediction can still count as a correct localization
in the shared shape-agnostic comparison below, but not in this same-class table.

## Reproducible comparison protocol

- Same 595 public development-validation images, losslessly saved after a single common
  320x240 resize. Source annotation hashes and every bundle file hash are retained.
- Shared task: **yellow goal localization**, ignoring shape. The baseline cannot classify
  shapes or identify orange goals with its frozen yellow-only setting. Those capabilities
  must be reported as unsupported, not inferred from a yellow blob.
- Both methods get exactly the same predecoded RGB arrays; order alternates AB/BA.
- One-to-one maximum-cardinality box matching at IoU >=0.5. Duplicate/unmatched boxes
  are false positives. Missed yellow boxes are false negatives. Baseline inclusive box
  endpoints are converted to the common half-open convention.
- Student output is filtered to accepted yellow classes for this shared task; its timed
  computation still searches/classifies both goal colors. Threshold stays at 0.8.
- Baseline parameters are unchanged. Its original live resolution is 540x540, but this
  controlled test uses 320x240 for both. The fixed 200-pixel filter is **not scaled**.
  Thus these results describe these configurations at 320x240, not optimized versions
  of each method or the baseline's original 540x540 live performance.
- Five warm-up frames; timed section excludes loading, capture, startup, drawing and
  transport. Report mean/median/p95 processing latency. Reciprocal mean is processing
  throughput, **not live camera FPS**. One OpenCV thread; one TFLite inference thread.
- Development scenes are correlated and have been reused during tuning. Upstream labels
  may be incomplete. Results are exploratory; an independent labeled Pi-camera recording
  is still required for a defensible final field comparison. Do not tune on final test data.
- Image source: Cheese, *Cats-and-Dogs*, Roboflow Universe V10, CC BY 4.0. Derived
  frames are resized and use the project's inferred grouped split. Retain the source and
  license attribution in `THIRD_PARTY_NOTICES.md` with any shared comparison data.
- Baseline color/size settings have not been recalibrated for these public images. A win
  here would not establish that learned detection is universally superior to color blobs.

## Commands on the Pi

The native adapter uses the already installed `libtensorflow-lite.so.2.20.0` through
the [TensorFlow Lite C API](https://github.com/tensorflow/tensorflow/blob/v2.20.0/tensorflow/lite/core/c/c_api.h).
No Python TensorFlow installation is needed. Bundle imports use its local `dtr` package.
Thirty golden crops compare labels and scores to the Mac export (score tolerance 0.03).

```bash
cd /home/pacman/pi-comparison-20261005
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 pi_compare.py \
  --golden-only --output golden-repeat.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 pi_compare.py \
  --output comparison-repeat.json
```

Use new output names; existing evidence is never overwritten. Reports include per-frame
boxes/timings in the accompanying `.frames.jsonl`. Default is all 595 frames; `--limit 20`
is a smoke test, not the complete accuracy evaluation.

For a bounded visual camera trial, place targets in view and run:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 pi_live_compare.py \
  --task goal --frames 100 --output camera-goal-trial-01
```

Outputs are raw PNGs, side-by-side annotated PNGs (baseline left, student right), and
timestamped observations. Saved pairs use the same captured frame. Camera format RGB888
is converted from its BGR memory order before inference. `--task balloon` exercises the
balloon student, but the baseline remains yellow-only, so that is not a like-for-like accuracy
trial. Neither script sends control commands. The combined sequential comparison loop is
not either method's independent live FPS. No ground-truth labels are inferred from live boxes.

Existing baseline's own visual demo, unchanged:

```bash
python3 /home/pacman/Documents/TESTING/demo.py \
  --source picamera --backend numpy --display web --port 8000 --frames 300
```

That demo binds HTTP on all Pi interfaces. Use only on a trusted test network. Do not run
two camera-owning programs together. The paired test saves files rather than starting a server.

## Evidence locations on the Mac

- `runs/{goal,balloon}-pi-student-20261005/`: model, metadata, provenance, learning curves, crop metrics.
- `runs/pi-students-frames-20261005.json`: per-class full-frame student development metrics.
- `runs/pi-comparison-20261005/`: frozen code/models/golden crops/595 comparison frames/checksums.
- `runs/pi-comparison-mac-20261005.json`: host-only sanity comparison; not Pi timing.
- `runs/pi-results-20261005/comparison-pi.json`: actual Pi metrics and runtime identity.
- `runs/pi-results-20261005/comparison-pi.frames.jsonl`: all 1,190 per-method/frame records.
- `runs/pi-results-20261005/golden-pi.json`: 30 cross-runtime reference comparisons.
- `runs/pi-results-20261005/camera-smoke/`: 20 live frame observations and saved image pairs.
- `tests/test_pi_comparison.py`: matching and native-buffer guards.

## Actual Pi result

All 595 frames completed on `vision`, original Pi Zero W / ARMv6, Python 3.13.5,
native TFLite 2.20.0, with 30 golden label/score checks passed. These figures are
from the Pi, not desktop estimates. Original baseline SHA256 remains
`f46b1dfde183e4695a9daa00ff8d3e9ca01a5ab1c85e969b9992b332a58ce8a7`.

| Shared yellow-localization task | Baseline RGB components | Distilled INT8 alternative |
| --- | ---: | ---: |
| True positives | 41 | 314 |
| False positives | 480 | 504 |
| False negatives | 492 | 219 |
| Precision | 7.87% | 38.39% |
| Recall | 7.69% | 58.91% |
| F1 | 7.78% | 46.48% |
| Mean processing latency | 71.32 ms | 761.43 ms |
| Median processing latency | 65.75 ms | 807.99 ms |
| p95 processing latency | 101.92 ms | 863.79 ms |
| Processing throughput (not live FPS) | 14.02 frames/s | 1.31 frames/s |

**Conclusion:** in this frozen 320x240 configuration, the alternative recovers substantially
more yellow targets and improves precision, but takes about 10.7 times longer per frame.
Neither configuration is accurate enough to claim reliable competition performance.
The alternative is now executable and measurable on the actual Pi, but the 12-candidate
configuration is slow. Candidate budgeting, rejection of hard negatives and goal shape
recognition are concrete next development priorities. The baseline needs a separately
recorded, train/calibration-only color/size calibration before making a broader algorithm claim.
INT8 reduced artifact size; this experiment does not measure a speedup versus FP32 on
the Pi, because FP32 inference was not benchmarked there.

Mac and Pi full-frame predictions are not completely identical (316 versus 314 shared-task
true positives). They use different TFLite versions/backends. The 30 golden crops passed
the declared label/score tolerance; that does not imply bit-identical predictions everywhere.
Use the Pi table for the hardware comparison. A cpuinfo topology warning appeared on the
Pi; both model verification and the complete comparison still finished successfully.

The paired camera smoke test completed 20 frames, with strictly increasing sensor timestamps.
Three same-frame pairs and raw PNGs were saved. This was an unstaged desk scene, not an
arena accuracy trial. The final pair shows an apparent orange-triangle false positive on
the desk; camera mounting/orientation and target/negative-scene testing still need attention.
It proves capture-to-detection plumbing only; it cannot establish field accuracy without labels.
After testing, the Pi reported no throttling flags (`0x0`) and 41.7 C. Original baseline
checksum remained unchanged. All camera-owning test processes exited.

Maximum absolute score difference in the 30 golden crops was 0.01984 (tolerance 0.03).
Shared comparison-process high-water RSS was 113,360 KiB; this is not a per-method memory
comparison. Local validation: 160 tests passed, 2 skipped; Ruff passed.
