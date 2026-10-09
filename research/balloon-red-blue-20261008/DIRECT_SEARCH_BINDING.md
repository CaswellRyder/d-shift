# Direct-address native search binding

2026-10-08. Exact implementation-cost experiment for the original ARMv6 Pi Zero W.
No retraining, threshold change, proposal-policy change, or flight qualification.

**Measured result: 5.8% higher processing throughput with identical full
detections in the paired Pi replay.** The optimization stays opt-in until its
live-camera integration is separately checked.

## Profile and change

A fresh instrumented profile of the existing `mser_fast` search on the fixed
16-scene replay took 4.960 seconds. Nested cumulative times are not additive:

- MSER detection: 2.206 seconds across 32 calls.
- Native `reduce` wrapper: 1.470 seconds cumulative across 816 calls.
- NumPy `data_as`: 0.383 seconds cumulative across 2,448 calls, including
  0.298 seconds in ctypes casts.
- Convex hull: 0.142 seconds across 606 calls.

The change avoids three NumPy/ctypes pointer-cast wrappers per region. An explicit
`direct=True` mode passes the validated array addresses to `c_void_p` parameters.
The Python argument arrays and fresh output array remain alive through the
synchronous native call. Dtype, contiguous shape, region-count, coordinate,
library hash, ABI, and return-size checks remain in place. Outputs are still
independently allocated, not reusable scratch memory exposed to callers.

The C source and compiled Pi library are unchanged. `direct=False` stays the
default. `mser_direct` is an opt-in research replay mode, not a production swap.
Model weights, MSER settings, scoring, selection order and suppression remain
unchanged. Neither camera capture nor ESP32/motor commands are involved.

## Parity and test boundaries

Host comparison: exact proposal dictionaries and full detections for both frozen
student seeds on all 16 scenes, with zero score difference from the existing
host golden. Both pointer modes pass randomized geometry, duplicate-point,
threshold, invalid-coordinate, buffer-contract, and independent-output tests.
The complete current-working-tree suite has 689 passes and 2 skips. Scoped Ruff
checks pass. Unrelated dirty files were preserved and not reviewed as this work.

The comparison tool requires at least two complete trials per method and rejects
changed model/bundle/runtime/golden/benchmark identities. Direct-address trials
must also use the same native library hash. It compares full detection
dictionaries across methods, while the underlying trial verifier checks repeated
frames, source order, hashes and counts. Repetitions are timing samples, not
independent accuracy examples. Existing blue misses/false positives remain.

## Hardware protocol and artifacts

Fresh isolated bundle:
`artifacts/balloon-pi-direct-search-20261008`, copied to the corresponding
`/home/pacman/balloon-pi-direct-search-20261008` directory. Earlier bundles remain
unchanged. Bundle SHA:
`9918b190f481e5682cbaf34986840f7e27fcad181ba0ac2c9820ae2952875ace`.

Pi native reducer SHA:
`d7620537d7e9850f0f794a26facc188063f289c6ad9f5d793693ad5715e43303`.
Native TFLite runtime SHA:
`54ed7fdda0b5b5f6db05a538eb800b7bc7ecd70d32f67dc13d1e78817d8b653a`.
Student seed42 SHA:
`a8267a9337adfe69a6e30f4e74a3a4a63fa12eb4710ec38a08ff3dd4f06f33a7`.

Sequential order: cast A, direct A, direct B, cast B. Each trial warms and checks
the 16 scenes, then times two complete rounds, with one OpenCV/inference thread.
Camera/file loading and golden-check overhead are excluded from per-frame
processing time. The Pi had no Python/camera workload at the preflight check;
ordinary system services were not disabled.

`direct-search-desktop-parity.json` records host parity.
Raw pre-change profile remains ignored at
`artifacts/balloon-fast-profile-20261008.json`; profile times are instrumented,
not a replacement throughput claim.
Profile artifact SHA:
`76e3587a25906f4eef7997a27424ab1d37e71097ca29280337b183f4c86b2ecd`.

## Completed hardware result

All four isolated trials finished. Each method has 64 timed processing samples
over the same 16 unique scenes. Pooled processing throughput increased from
**3.0187 to 3.1950 FPS**, a **1.0584x** ratio. Mean latency fell from
**331.274 to 312.992 ms** (5.52% lower); pooled p95 fell **11.75%**.
Mean search time fell from **223.231 to 204.599 ms**. Neural calls remained
5.625 per frame and inference stayed approximately 107 ms/frame.

Full detection dictionaries match exactly across both methods, trials and
repetitions, including boxes, proposal scores, neural probabilities and
suppression. Maximum difference from the desktop golden was less than 0.000001.
Throttle flags stayed `0x0`; maximum direct-mode RSS was 114,784 KiB (112.09 MiB).
No remaining Python/camera process was observed after the trials. No parallel
model, capture or profile ran during the timed comparisons.

The comparator is the **already optimized** `mser_fast`, not the original slow
MSER or the other team's pixel-only method. Do not multiply this result into an
unmeasured live-camera FPS claim. The original/indoor accuracy counts remain
unchanged, including the seed42 indoor blue 9 TP / 4 FP / 3 FN. The 94% precision
and recall requirement is still unmet.

Hash-bound reports and pooled results:
`direct-search-pi-measurements.json`. Ignored raw logs:
`artifacts/balloon-pi-direct-results-20261008`. Next useful optimization target is
MSER detection itself; changing its policy will need explicit recall checks,
not merely faster background-frame timing.

Reproduce a fresh isolated bundle with `scripts.build_fast_balloon_pi_bundle`,
compile its native helper on the Pi, and run `pi_balloon_search_bench.py` with
`--search mser_fast` or `--search mser_direct`, the same region/runtime libraries,
and unique output filenames. Use the sequential trial order above.

Summarize all four copied trial logs with:

```sh
.venv/bin/python -m scripts.compare_exact_balloon_search --base artifacts/balloon-pi-direct-search-20261008 --results artifacts/balloon-pi-direct-results-20261008 --methods mser_fast mser_direct --output artifacts/fresh-direct-comparison.json
```
