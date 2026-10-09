# Exact MSER search optimization on original Pi Zero W

2026-10-08. **Measured 28.4% processing-throughput improvement without changing
any detection in the paired development replay.** No retraining, threshold
change or deployment promotion.

## What changed

Research-only `mser_fast` keeps the original MSER parameters, opponent-color
planes, proposal scores, 12-proposal budget, student weights, classification
threshold and suppression logic. It changes implementation cost:

1. Split RGB channels once, avoiding repeated copies of interleaved slices.
2. In a bounded native C pass over each region, sum color values, find its bounds
   and retain only each row's minimum/maximum x coordinates.
3. Reject the same low-contrast or impossible bounding geometries before hull
   calculation, then run the existing candidate/scoring function on the reduced
   points.

Every discarded point lies on the segment joining its row's retained extrema,
so the convex hull is preserved. The color test uses `sum < 15 * count`, exactly
equivalent to the original uint8 mean threshold; duplicated points retain their
original weighting. Bounds and aspect-ratio checks retain the original inclusive
geometry. Model inputs and search semantics stay unchanged.

The C ABI is limited to contiguous 320x240 uint8 planes and at most 76,800 int32
coordinate pairs. Coordinates are checked before indexing, output is bounded to
480 points, and Python checks buffer contracts and library hash/ABI. Invalid
inputs fail rather than silently switching implementations. The library is
built into the separate research bundle, not installed system-wide.

## Paired hardware result

Original Raspberry Pi Zero W Rev 1.1, ARMv6, same existing native TFLite runtime,
one OpenCV/inference thread, student `new-views-42`. Sequence: old A, optimized A,
optimized B, old B. One vision process at a time. Desktop/system services stayed
enabled. One read-only process check occurred during optimized A startup;
the final process check found no remaining benchmark process.

- Old MSER: **2.329 FPS**, mean **429.380 ms**, pooled p95 **826.977 ms**.
- Optimized MSER: **2.991 FPS**, mean **334.335 ms**, pooled p95 **643.001 ms**.
- Mean search: **320.586 → 225.304 ms** (29.7% less search time).
- Throughput: **1.2843x**; mean latency **22.1% lower**, p95 **22.2% lower**.
- Neural calls unchanged at 5.625/frame; mean neural time remains about 108 ms.
- Maximum optimized process RSS: 115,152 KiB (112.45 MiB).
- Recorded throttle flags all `0x0`; temperatures approximately 43.3–46.5 C.

Each method pools 64 timed frames (two repeats of 16 scenes in each of two
trials). This is **16 unique development scenes**, not 64 accuracy examples.
Percentiles are recomputed from raw records. Timings exclude camera capture,
model/file loading, warmup and golden-comparison overhead. They are not live FPS.
The comparator is the same neural model with old MSER, not another team's
pixel-only implementation. This paired result is preferable to comparing against
an older run on a different hardware workload or scene mix.

## Exactness and accuracy

- Desktop: exact proposal dictionaries for all 16 scenes; both seed42 and seed43
  full-scene predictions match the existing desktop golden with zero score delta.
- Pi seed42: all warmup/timed frames match the original desktop golden, maximum
  neural score difference below 0.000001.
- Offline comparison of all four Pi logs: **full detection dictionaries are
  identical across variants and repeats**, including proposal area/score, crop,
  probabilities, labels and suppression. The comparison rejects identity changes
  or missing repeats instead of comparing unrelated trials.
- Unit coverage includes 300 randomized point clouds, duplicated points,
  threshold boundaries, extreme/invalid coordinates, maximum output size,
  generated full scenes, buffer contracts and hash rejection.

This preserves weaknesses as well as strengths. The indoor panel still has blue
TP/FP/FN 9/4/3; the original panel still has two red false positives. No claim of
94% overall precision/recall or flight readiness follows. Reserved final-test
data remain untouched. No camera, ESP32, serial or motor commands were used.

## Evidence and reproduction

Implementation checkpoint: `27b90a56`.

- `fast-search-desktop-parity.json`: host proposal/model parity and native build.
- `fast-search-pi-measurements.json`: reports, pooled timings, scene counts,
  identities, health, build receipt and exact comparison.
- Ignored raw logs: `artifacts/balloon-pi-fast-search-results-20261008`.
- New isolated bundle: `artifacts/balloon-pi-fast-search-20261008`, copied to
  `/home/pacman/balloon-pi-fast-search-20261008`. Earlier bundles are unchanged.
- Bundle SHA-256:
  `782f7715f552266478968dafcb85539e8422574fdca6efbdf8b3bff7d7c3ef01`.
- Pi native helper SHA-256:
  `d7620537d7e9850f0f794a26facc188063f289c6ad9f5d793693ad5715e43303`.

Build a fresh bundle with `scripts.build_fast_balloon_pi_bundle`, using the
previous verified current-model bundle as `--reference`. Compile
`build_native_balloon_regions.py` on the Pi, then run the existing replay command
with `--search mser_fast --region-library native/regions.so`. For controls use
`--search mser` and omit the region library. Keep the runtime/thread environment
from `CURRENT_PI_SEARCH.md`, sequential execution and fresh output filenames.

Reproduce the comparison from the host:

```sh
.venv/bin/python -m scripts.compare_exact_balloon_search --base artifacts/balloon-pi-fast-search-20261008 --results artifacts/balloon-pi-fast-search-results-20261008 --output artifacts/fresh-exact-search-summary.json
```

Next work: measure this research path with real camera capture, correct 180-degree
orientation and frame freshness; keep that timing distinct from labeled field
accuracy. Blue clutter/part errors still need further model/data work. The
optimization remains opt-in research code, not an unqualified production swap.
