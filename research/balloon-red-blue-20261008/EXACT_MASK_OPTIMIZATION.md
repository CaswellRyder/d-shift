# Exact red/blue mask acceleration — 2026-10-08

Status: opt-in research acceleration. No color thresholds, model weights,
proposal selection rules, confidence threshold, or production defaults changed.
This cannot repair existing detection errors or satisfy the 94% gate by itself.

## Evidence and rejected approach

The Pi profiler found work distributed across HSV masking, contour extraction,
and Python candidate handling. In an instrumented 20-frame profile, `findContours`
used 0.713 seconds, masks 0.546 seconds, morphology 0.211 seconds, and the remainder
included proposal filtering/ranking. Profiling overhead makes that instrumented
total unsuitable as the normal frame-time estimate.

An exact RGB24 membership table stores red in bit 0 and blue in bit 1 for every
possible 8-bit RGB color. It is 16 MiB and preserves the existing HSV rules.
The first implementation used NumPy indexing and intermediate index buffers.
On the actual Pi it was **slower** than OpenCV and was rejected as a performance
choice. It remains an explicit comparison backend, not the default.

The second implementation is a small, locally compiled C loop with checked ctypes
buffer contracts. It directly reads RGB pixels, looks up membership, and emits
the two masks. No downloaded native binary, new system package, binary neural
network kernel, or Samsung code is involved. The existing system library is not
replaced. C source, compiler flags, and library checksum are recorded.

## Exactness and isolated search timing

- All 16,777,216 RGB colors matched OpenCV 4.11.0 on the Mac and OpenCV 4.10.0 on
  the actual ARMv6 Pi. Both NumPy and native paths were verified on the Pi.
- Unit tests also cover strided/channel-reversed input, invalid dtypes/tables,
  incorrect taxonomy contracts, and requested-but-missing lookup files.
- Four fixed development photos, ten timed rounds after warmup, alternating
  backend order. Algorithms execute sequentially, never concurrently.
- Every compared proposal list must match the HSV reference exactly, including
  order, boxes, crop boxes, area, color group and score. The native comparison
  checks 132 lists including warmups.

| Actual Pi backend | Mask mean | Total proposal-search mean |
| --- | ---: | ---: |
| OpenCV HSV | 27.62 ms | 100.07 ms |
| NumPy lookup — rejected | 35.60 ms | 107.94 ms |
| Native lookup | 15.50 ms | 89.09 ms |

The native lookup cuts mask latency about 44% and total proposal latency about
11%. It does **not** imply a 44% whole-system speedup: contour processing and
neural inference remain unchanged. Tables/runtime initialization are excluded
from warmed timings and are separately recorded by the full-frame harness.

## Whole-frame check and decision

Same context FP32 weights, same 12-crop budget, four development photos repeated
five times per run. Fresh processes run sequentially in HSV A / native A /
native B / HSV B order on the same rebuilt TFLite library. No competing vision
process was started. These are processing-only timings, not live FPS.

| Backend | Mean frame ms A / B | Pooled processing FPS |
| --- | ---: | ---: |
| HSV | 319.19 / 322.95 | 3.11 |
| Native lookup | 316.07 / 310.83 | 3.19 |

The measured whole-frame throughput gain is only about **2.4%** in this small
paired replay, not 44%. Peak process RSS rises from about 112 MiB to 135 MiB;
table validation/loading and first mask take 1.15–1.17 seconds versus about 2 ms
for the baseline. No thermal/undervoltage flags were reported. Development
counts remain exactly red TP/FP/FN 2/3/0 and blue 1/0/2 for every run; repeated
photos are never counted as additional accuracy trials.
All non-timing observation fields also matched exactly across the four 20-frame
logs, not just their aggregate counts.

**Decision:** keep HSV as default. Native lookup remains a bounded opt-in option,
not a recommended default given the modest measured full-frame benefit and
memory/startup cost. Future optimization should consider a fused small-table
integer HSV implementation or reducing contour/Python overhead, with the same
exactness checks, rather than assuming weight compression solves the pipeline.
Accuracy work remains higher priority: representative data, blue proposal recall,
and red non-balloon rejection have not reached the requested target.

Receipts: `lookup-mac-verification.json`, `lookup-pi-verification.json`,
`native-lookup-pi-verification.json`, `native-color-lookup.json`,
`native-search-comparison.json`, `native-mask-bundle.json`, and
`native-mask-replay.json`. Original raw replay logs remain in ignored
`artifacts/red-blue-pi-native-results-20261008`. Nothing was installed globally.

Lookup SHA256:
`3817ed4fd607bb43586adc77f5835a2a9331ed76cd81fc52b3668f99c192e44c`

ARMv6 native library SHA256:
`5507aa6453baaad4cc2aa2f27f9d1fc3af1f3afad113ae9235af8877eaa12708`

## Reproduction and operational boundaries

Build a new table without opening any dataset images:

```sh
.venv/bin/python -m scripts.build_red_blue_lookup \
  --output artifacts/red-blue-lookup-20261008/colors.bin
```

Add it to a fresh Pi bundle using the existing builder's `--red-blue-lookup`
argument. On the Pi, compile the bundled source into the bundle directory:

```sh
python3 build_native_color_lookup.py \
  --source dtr/native_color_lookup.c --output native-color-lookup.so
python3 build_red_blue_lookup.py --output red-blue-lookup.bin --verify \
  --native-library native-color-lookup.so --receipt native-verification.json
python3 pi_red_blue_search_bench.py --base . \
  --native-library native-color-lookup.so --output search-comparison.json
```

Use `pi_red_blue_bench.py --mask-backend native` for isolated replay/camera
experiments; `hsv` remains its default. Outside this harness, the research runtime
requires both `DTR_RED_BLUE_COLOR_LOOKUP` and `DTR_RED_BLUE_LOOKUP_LIBRARY` to use
the native path. Setting only the table selects the slower NumPy implementation.
Missing/invalid explicit configuration fails rather than silently falling back.

The table's versioned contract is tied to the current HSV bands. If the bands
change, regenerate/reverify the table and version the contract appropriately.
Keep native-library and table hashes with every run. The extra 16 MiB table and
initialization cost must be considered when combining goal and balloon pipelines.
No live-camera speed claim is made for this mask change; the previous camera view
was obstructed, so representative target recordings remain required.
