# D-SHIFT v1.0.0 model card and weight lineage

**Status: versioned research/integration baseline, not flight-qualified.**
This snapshot packages existing trained models. No new training, threshold tuning,
Pi deployment, motor operation, or independent accuracy evaluation is implied.
The package version and original Python distribution version are separate.

## Selected artifacts

| Role | Artifact | Choice and scope |
| --- | --- | --- |
| Goal inference | `models/v1.0.0/goal/student.float.tflite` | Context-16K, 16,183 parameters; primary goal choice |
| Goal quantization comparison | `models/v1.0.0/goal/student.int8.tflite` | Same selected Keras checkpoint, INT8 export; slower on tested ARMv6 runtime |
| Goal training checkpoint | `models/v1.0.0/goal/student.keras` | Selected context CNN, not MobileNetV4 architecture |
| Balloon inference | `models/v1.0.0/balloon/student.int8.tflite` | Earlier 6,131-parameter baseline; separate validation limits |
| Balloon training checkpoint | `models/v1.0.0/balloon/student.keras` | Three strided convolutions and global pooling |
| Pi runtime | Release asset `d-shift-v1.0.0-armv6-runtime.zip` | Isolated TensorFlow Lite 2.20.0 C API build for ARMv6/VFPv2 |

Each TFLite file requires its adjacent, same-stem JSON metadata. Metadata specifies
ordered classes, 64x64 RGB input, resize/normalization, quantization, provisional
threshold, hashes, training inputs, and unapproved status. Do not mix metadata
between exports. The goal FP32 file is 68,064 bytes with SHA-256
`147671bbff83a8d0f57ff01a0fc8836441790b7b5c919f0e1ade1bb16c75fea3`.

## Start-to-end weight lineage

```text
timm ImageNet MobileNetV4-Conv-Small checkpoint
  → Keras import with numerical parity check (shared backbone)
    ├─ initial DTR goal teacher
    │   → reviewed proposal-adapted goal teacher
    │     → Context-16K distilled student.keras
    │       ├─ FP32 TFLite (selected)
    │       └─ calibrated INT8 TFLite (comparison)
    └─ initial DTR balloon teacher
        → reviewed proposal-adapted balloon teacher
          → Tiny-6K distilled student.keras
            └─ calibrated INT8 TFLite (companion baseline)
```

The weight-lineage release ZIP preserves these original paths:

1. `artifacts/pretrained/mobilenetv4_conv_small.{keras,json}`: imported
   `mobilenetv4_conv_small.e2400_r224_in1k`, with original import/parity metadata.
   This is the Keras-imported starting checkpoint, not a second copy of the raw timm file.
2. `runs/{goal,balloon}-dtr-v10-20261003/teacher.{keras,json}`: initial real-DTR
   fine-tuned teachers, with configs, provenance, and head/fine-tuning epoch logs.
3. `runs/{goal,balloon}-proposals-20261005/teacher.{keras,json}`: teachers adapted
   to reviewed proposal crops and negatives, with the same provenance/logs.
4. `runs/student-research-20261005/context-kd/`: selected goal Keras/FP32/INT8
   artifacts, metadata, training provenance, epochs, and validation summary.
5. `runs/balloon-pi-student-20261005/`: selected balloon Keras/INT8 artifacts,
   metadata, provenance, epochs, and validation summary.

`models/v1.0.0/lineage.json` verifies actual parent weight hashes, not just names.
Original metadata is copied without rewriting historical flags or absolute paths.
Those paths identify the training machine; inference uses the local artifact path.
Historical `pi_zero_verified: false` fields predate subsequent Pi measurements;
later hardware evidence is stored separately rather than retroactively editing them.

The goal student uses 25 distillation epochs, teacher temperature 4, equal hard-label
and temperature-scaled KL weighting, Adam 0.001, seed 42, no augmentation, and
validation-hard-loss checkpoint selection. Balloon uses eight epochs with the same
distillation temperature/loss settings. INT8 calibration uses 500 training crops.
See each task's provenance/config and epoch logs for the recorded execution.

The cached goal teacher logits originated in `runs/goal-pi-student-long-20261005`.
Their checksum is recorded in provenance. They are data-derived training inputs,
not model weights, and are deliberately excluded; regenerate from the exact
adapted teacher and exact reviewed manifest if reproducing training.

## Data and reproducibility boundaries

Source: Cheese's Cats-and-Dogs DTR Roboflow V10, CC BY 4.0, as described in
[data sources](DATA_SOURCES.md) and [third-party notices](../THIRD_PARTY_NOTICES.md).
Dataset images, crops, COCO annotations, per-sample manifests, cached logits,
reserved test pixels, and recordings are not uploaded with this release.
Dataset hashes, reviewed label/negative configs, and provenance remain documented.

This is a complete selected-weight lineage, **not an archive of every optimization
step or a bit-for-bit resumable training session**. Intermediate epoch checkpoints,
optimizer/RNG snapshots, synthetic smoke weights, rejected architecture/refinement
runs, and the separate YOLO detector are not selected v1 artifacts. Their work
remains documented in [project history](PROJECT_HISTORY.md). Original source hashes
in training provenance may differ from the current release implementation.
Re-running training requires the original data splits/reviews and environment;
identical results are not guaranteed merely by possessing the checkpoints.

## Frozen inference configuration

See [`configs/releases/v1.0.0.json`](../configs/releases/v1.0.0.json): goal search
320x240, RGB conversion, mounted-camera rotation 180 degrees, up to 12 candidates,
four classifications per temporal call, 0.5 s scans, 0.6 s refresh, 1.8 s quiet
background refresh, 1.0 s class expiry, and native tracking differences.
Threshold remains 0.8. Shape-only/color-unknown evidence is not a confirmed target.
Rotation corrects mounting, not camera calibration or world heading.

The profile records the selected configuration; existing viewers are not silently
switched to these models. Most recent performance tests exercise the goal branch.
There is no verified combined goal/balloon scheduler, metric range estimate,
possession confirmation, goal-opening clearance, or flight-controller integration.

## Download and verify

Authorized GitHub access is required because the repository and release are private.
From a fresh clone checked out at tag `v1.0.0`:

```bash
python3 scripts/verify_release.py
gh release download v1.0.0 --repo CaswellRyder/d-shift --dir output/v1-download
python3 scripts/verify_release.py --assets output/v1-download
```

The verifier checks committed files/source hashes, both ZIP checksums, and every
archived member against the committed manifest. `SHA256SUMS` is also downloadable.
Use a new extraction directory; the archives contain historical paths and should
not overwrite existing local research runs. Small student models need no release
download: they are already in Git. The larger lineage weights and runtime do.

On the Mac with the documented project environment:

```bash
PYTHONPATH=src TF_CPP_MIN_LOG_LEVEL=3 .venv/bin/python scripts/verify_release.py --smoke
```

This runs generated numerical probes through both Keras students and all three
TFLite exports. It checks finite scores and FP32 export parity, not recognition
accuracy or physical-device speed. No dataset or camera is opened.

For a user-provided image on the Mac or Pi with its inference dependencies:

```bash
PYTHONPATH=src python3 -m dtr.cli predict \
  --model models/v1.0.0/goal/student.float.tflite \
  --image /path/to/goal-crop.png --allow-unvalidated
```

On ARMv6, extract the runtime asset into a separate directory and set
`DTR_TFLITE_LIBRARY` to its absolute `libtensorflowlite_c.candidate.so` path.
The override is used by the native C API fallback when `tflite_runtime` and
TensorFlow Python are absent. Do not replace `/usr/lib` or install desktop
TensorFlow on the Pi. See [Pi dependencies](PI.md) and [runtime build](TFLITE_REBUILD.md).
The runtime is a Linux ARMv6 binary and cannot execute on the Mac.

## Evidence and known limits

- Goal crop accuracy: 98.31%, including background, on reused development crops.
- Six-class full-frame goal F1: 63.23% on 595 reused development frames.
- Yellow-only localization: precision 88.64%, recall 90.81%, F1 89.71% on those
  common-input Pi frames. This ignores shape and is not six-class accuracy.
- Context FP32 crop latency: 23.64 ms with rebuilt runtime vs 63.82 ms installed.
- Delivered generated-paced replay: 7.93 FPS; actual unlabeled desk camera: 3.31 FPS.
- Balloon baseline: 99.41% crop accuracy, but earlier full-frame green precision/
  recall 58.66%/79.48% and purple 22.25%/96.26%. Strong crops do not qualify capture.

The current camera orientation fix occurred after the complete benchmark. It
received smoke checks, not a complete new performance evaluation. Live desk
footage had no staged targets. The public development scenes were repeatedly
used for selection and are not independent competition recordings. Small/orange
goal proposal recall and wrong-target rejection remain important limitations.

Evidence summaries are committed under `models/v1.0.0/evidence/`; detailed scope
is in [comparison](COMPETITION_COMPARISON.md), [student research](STUDENT_RESEARCH.md),
and [Pi comparison](PI_COMPARISON.md). Pixel benchmark numbers are retained as
evaluation evidence, but the other implementation's source is not included.

## Version policy and licensing

`v1.0.0` is a research prerelease on GitHub, freezing the planned integration
baseline. It does not change any `deployment_approved` field. New selected weights,
preprocessing/class changes, or runtime choices require a new manifest/version;
never replace v1 assets in place. Git's tag pins the source and documentation.

Retain the project Apache-2.0 license, MobileNetV4/timm attribution, and dataset
CC BY 4.0 attribution. The runtime ZIP includes its upstream dependency notices.
The optional YOLO branch is outside the selected weight lineage and retains its
separate licensing; no YOLO weights are redistributed here.
