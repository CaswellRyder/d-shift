# Isolated TensorFlow Lite ARMv6 rebuild

Started 2026-10-06. This is a runtime experiment, not model retraining. The system
library, original Pi demo, model weights and web viewer must remain unchanged.

## Build identity

- Target: original Raspberry Pi Zero W, ARM1176JZF-S / ARMv6 / VFPv2, hard-float Linux.
- TensorFlow tag: `v2.20.0`; commit `72fbba3d20f4616d7312b5e2b7f79daf6e82f2fa`.
- C API target: `tensorflowlite_c`; no Python TensorFlow wheel is required.
- Cross toolchain: Zig 0.14.1's Clang 19.1.7, `arm-linux-gnueabihf.2.28`,
  `-mcpu=arm1176jzf_s`, `-O3`, `-fno-fast-math`, PIC. Bundled C++ runtime;
  glibc symbol compatibility target 2.28 (Pi has glibc 2.41).
- CMake 3.31.6 and Ninja 1.11.1.3. Native `flatc` built from TensorFlow's pinned
  FlatBuffers dependency with AppleClang 21.0.0.
- XNNPACK, RUY dispatch, GPU, NNAPI and external delegates disabled. Resource
  support retained because registered CPU operators require its symbols.
  This is a scalar CPU research build; it cannot add NEON instructions to ARMv6.
- Build tree: `/Volumes/HDD_Storage_Extended/dtr-tflite-rebuild-20261006`.
  Build tools are isolated in that directory, not installed into the project venv.

The installed library already reports ARMv6/VFPv2 attributes. This is not a fix
for an ARMv7 binary mistakenly installed on the Pi. Performance, if improved,
must be measured; compiler and runtime differences are confounded in this trial.

## Attempts and scoped repairs

1. Two isolated Apple Container Linux builders stalled before startup. Their
   launch processes were stopped. Existing unrelated containers were left alone.
   A small Pi sysroot archive was copied for inspection, but the successful smoke
   executable uses the cross-toolchain's bundled libc compatibility support instead.
2. Native macOS cross-compilation produced `armv6-smoke`. It ran on the actual Pi:
   `ARMv6 C++ smoke: atomic=7 magnitude=5.0`. ELF attributes identify ARMv6/VFPv2.
3. Initial TensorFlow configuration failed because CMake unconditionally applies
   options to `xnnpack-delegate`, even with XNNPACK off. The recorded
   `scripts/tflite_rebuild/xnnpack-disabled.patch` guards the absent target only.
4. Compilation then failed because CMake did not supply the version definitions
   normally provided by Bazel. `release-defines.h` supplies 2.20.0 and an empty
   suffix, matching the pinned `tensorflow/tf_version.bzl`. It changes no kernels.
5. CPU embedding kernels require FP16 conversion headers even with delegates
   disabled. `fp16-cpu-headers.patch` enables the existing pinned header dependency.
   An initial patch-verification hunk failed; it was corrected before continuing.
6. Attempt 5 completed all 506 remaining compile/link steps and produced
   `build-armv6/libtensorflowlite_c.so`. Runtime/accuracy checks are separate.
   Its Pi load failed with an unresolved resource-variable function: disabling
   resource support was incompatible with the default registered kernel set.
7. Attempt 6 restored resource support and added `--no-undefined` at link time.
   It linked successfully with debug information stripped by the linker, producing
   a 3.2 MiB candidate. The failed attempt-5 binary and receipt are retained.

Logs `build-attempt-1.log` through `build-attempt-6.log` are retained in the
external build tree. Build, Pi loading and reference inference now pass;
those are separate checks, not a flight qualification.

## Reproduction

The external tree contains a Python 3.11 tools venv with pinned `cmake`, `ninja`
and `ziglang` packages. `tensorflow/` is the pinned clone; `host-tools/bin/flatc`
is built using TensorFlow's `tensorflow/lite/tools/cmake/native_tools/flatbuffers`
project. Source wrappers are in `scripts/tflite_rebuild/`.

```sh
bash scripts/tflite_rebuild/build.sh \
  /Volumes/HDD_Storage_Extended/dtr-tflite-rebuild-20261006
```

The recipe checks the source revision, checks/applies only the recorded build
patches, and builds a separate library. It does not install, overwrite the system
runtime, or configure startup services. Retain the external drive for rebuilding.

Primary build references: [TensorFlow's CMake guide](https://github.com/tensorflow/tensorflow/blob/v2.20.0/tensorflow/lite/g3doc/guide/build_cmake.md),
[C API build target](https://github.com/tensorflow/tensorflow/blob/v2.20.0/tensorflow/lite/c/CMakeLists.txt),
and [Zig cross-compilation overview](https://ziglang.org/learn/overview/).

## Runtime comparison contract

The installed library's initial SHA-256 is
`44c6aaeda9f650b59f13e1ac6816ecd21daf4d53abfb1d023619e68724e791bc`.
Pi artifacts belong only under `/home/pacman/dtr-runtime-build-20261006/`.
Candidate selection uses the existing `DTR_TFLITE_LIBRARY` environment override,
not `ldconfig`, system paths, package replacement or a global environment change.

Compare the same frozen six models (context, separable, warm; FP32 and INT8) and
21 reference crops in `pi-cv-progress-20261005`, using its checksum-verifying
`pi_research_crops.py`. Three rounds yield 63 timed calls per model after five
warmups. Labels and all scores must pass the existing parity tolerances before
timings can support a runtime choice. Imports, model startup, camera capture,
proposal search and communication are excluded from crop timings.

No camera or flight commands are part of this build experiment.

## Actual Pi crop results

Order: installed A, rebuilt A, rebuilt B, installed B; each run is a fresh Python
process, with the same frozen model/crop bundle and one inference thread. Values
below average the two 63-call means per runtime. This is a small reference set,
not a new independent model-accuracy evaluation.

| Model | Installed ms/crop | Rebuilt ms/crop | Throughput ratio |
| --- | ---: | ---: | ---: |
| Context FP32 | 63.82 | 23.64 | 2.70x |
| Context INT8 | 85.77 | 51.13 | 1.68x |
| Separable FP32 | 43.61 | 18.15 | 2.40x |
| Separable INT8 | 61.92 | 38.23 | 1.62x |
| Warm separable FP32 | 43.47 | 17.68 | 2.46x |
| Warm separable INT8 | 60.34 | 38.49 | 1.57x |

Every label matched the Mac references in all four runs. Maximum rebuilt FP32
score error was 9.54e-7 (tolerance 0.001); INT8 maximum was 0.00522 (tolerance
0.03), also present with the installed library. The candidate is not numerically
bit-identical. **FP32 remains faster than INT8** for these models on this board.

The library successfully loaded on the Pi and reported 2.20.0. Its ELF attributes
show ARM1176JZF-S, ARMv6, VFPv2 and VFP-register argument passing. Final file size
is 3,361,916 bytes. No separately installed C++ runtime is required by its dynamic
dependency list. The existing cpuinfo topology warning still appears; it did not
prevent inference or parity checks.

## Full context-model replay

Two runs per runtime, in installed / rebuilt / rebuilt / installed order. Same
240 translated development frames, context FP32 weights, native tracking-difference
backend, four-crop budget, label lifetime, source code and proposal policy.

| Runtime | Mean ms, A / B | Processing FPS, A / B | p95 ms, A / B |
| --- | --- | --- | --- |
| Installed | 185.06 / 184.12 | 5.40 / 5.43 | 411.96 / 418.66 |
| Rebuilt | 110.38 / 111.02 | 9.06 / 9.01 | 258.29 / 259.42 |

This is approximately **67% higher processing throughput**, not a 2.7x whole
pipeline speedup. Search and tracking still cost time. Each run made 439 neural
calls and 78 full scans. Rebuilt still exceeded 100 ms on 120–122 of 240 frames:
there is no guaranteed 10 Hz deadline. Both libraries emitted zero accepted
detections on the 24 black missing-target frames and recovered at least one
correct class on the first returned frame in 11/12 clips.

Six-class replay counts were installed TP/FP/FN 492/68/498 versus rebuilt
498/68/492. Faster execution can keep cached classifications inside the unchanged
one-second lifetime; this difference is not evidence that retraining improved
model accuracy. A first-pair per-frame audit found six additional accepted rows
and no removed observations. Those rows have rebuilt ages 856–865 ms; substituting
the installed runtime's processing duration gives 1,021–1,044 ms, above the same
1,000 ms expiry threshold. Camera capture, generation/decoding, rendering, transport, startup
and output writes are excluded from the reported processing rates. The nominal
replay clock advances 100 ms per frame independently of actual runtime.

The second run of each runtime also recorded **whole-process** peak RSS:
installed 113,448 KiB (110.8 MiB), rebuilt 110,864 KiB (108.3 MiB). This includes
imports, validation, decoding and stored replay records, not incremental library
memory or live-camera buffering. Both reported zero major page faults. Final board
temperature was 37.9 C; throttling `0x0`; 271 MiB available and 87 MiB swap occupied.
Swap occupancy alone does not show that these measured processes were paging.

## Using the candidate for further tests

The system library remains the default. No startup service or original Pi demo
was modified. To repeat the isolated candidate's replay on the Pi:

```sh
DTR_TFLITE_LIBRARY=/home/pacman/dtr-runtime-build-20261006/libtensorflowlite_c.candidate.so \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python3 /home/pacman/pi-native-tracking-20261006/pi_temporal_bench.py \
  --base /home/pacman/pi-comparison-20261005 \
  --model /home/pacman/pi-native-tracking-20261006/models/context.tflite \
  --policy temporal --difference-backend native \
  --output /home/pacman/dtr-runtime-build-20261006/my-next-replay.json
```

Use a fresh output filename. Do not copy the candidate over `/usr/lib`. It is a
promising integration-test runtime, not a general qualification of every TFLite
operator or a live-camera/flight release. The original context model can now be
tested at higher throughput without switching to the weaker separable candidate.

Project results, candidate, recipe and retained licenses are under
`runs/tflite-rebuild-20261006/`. Build logs and source/dependency trees remain on
the external drive. Local verification: 231 Python tests passed, two skipped;
Ruff and shell syntax checks passed. These checks are additional to actual Pi
inference and do not substitute for it.

After testing, the installed runtime retained its original SHA-256, and the
original demo remained `f46b1dfde183e4695a9daa00ff8d3e9ca01a5ab1c85e969b9992b332a58ce8a7`.
All 640 original comparison files still matched their checksums. Downloaded
result checksums matched the Pi copies. SSH was closed after the audit.

The final build receipt records compiler-input hashes, dependency Git revisions,
source patches and output SHA-256. Candidate library:
`54ed7fdda0b5b5f6db05a538eb800b7bc7ecd70d32f67dc13d1e78817d8b653a`.
Both failed temporary container definitions were removed; their external build
files are preserved and the unrelated running containers were not restarted.
