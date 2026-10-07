# D-SHIFT

**Distilled Spatial-context Hybrid Inference with Freshness-aware Tracking**

Hybrid computer vision for resource-constrained Defend the Republic (DTR) blimps.
D-SHIFT combines OpenCV region detection, a custom distilled neural classifier,
and bounded temporal tracking on the **original Raspberry Pi Zero W (ARMv6)**.

**Research software, not flight-qualified.** Publishing the code is separate from
approving autonomous operation. This project provides perception, not motor
control, navigation, verified distance, or capture confirmation.

[Documentation](docs/README.md) · [Model card](docs/MODEL_V1.md) ·
[Weights](https://github.com/CaswellRyder/d-shift/releases/tag/v1.0.0) ·
[Contributing](CONTRIBUTING.md) · [Security](SECURITY.md)

## How it works

```text
Camera → color/region proposals → distilled crop classifier
       → tracked, freshness-bounded observations → downstream mission software
```

- **Distillation:** offboard MobileNetV4-Conv-Small teachers train custom students.
- **Spatial context:** the goal CNN preserves a 2×2 layout and adds a context convolution.
- **Hybrid detection:** OpenCV proposes regions; neural inference classifies them.
- **Recognition:** orange/yellow circle, square, and triangle, plus background.
- **Tracking:** position updates, classification budgets, refresh intervals, and stale-label expiry.
- **Embedded inference:** an isolated ARMv6 TensorFlow Lite C API runtime.

The selected goal model is **Context-16K (16,183 parameters)**, not MobileNetV4's
architecture. A separate **Tiny-6K balloon baseline** recognizes green/purple
balloons but retains substantial full-frame false positives. Combined mission
operation is not qualified.

## Quick start

Use **Python 3.11** and [uv](https://docs.astral.sh/uv/) on a supported desktop
host. Access is restricted while public-release preparation is underway.

```bash
git clone https://github.com/CaswellRyder/d-shift.git
cd d-shift
uv sync --locked --extra dev
python3 scripts/verify_release.py
uv run --locked python scripts/verify_release.py --smoke
```

Generated numerical inputs verify loading and FP32 parity, not recognition
accuracy. Small selected weights are committed; this check needs no dataset,
teacher download, camera, or flight hardware.

Predict a crop you supply:

```bash
uv run --locked dtr --cpu predict \
  --model models/v1.0.0/goal/student.float.tflite \
  --image /path/to/goal-crop.png --allow-unvalidated
```

Desktop webcam/image viewer using released students:

```bash
uv run --locked python webcam_app.py --allow-unvalidated \
  --goal-model models/v1.0.0/goal/student.float.tflite \
  --balloon-model models/v1.0.0/balloon/student.int8.tflite
```

The viewer runs on localhost with browser camera permission. It is not an
Internet service and does not automatically use the Pi temporal scheduler.
See [visual testing](docs/TESTING.md).

**Do not install desktop TensorFlow on the Pi Zero.** Follow the [Pi guide](docs/PI.md)
and [model card](docs/MODEL_V1.md) for the separate ARMv6 runtime. Optional
teacher-import/Metal setup is in the [development guide](docs/DEVELOPMENT_GUIDE.md).

## Evidence and limitations

| Measurement | Result | Scope |
| --- | ---: | --- |
| Goal crop accuracy | 98.31% | Reused development crops, including background |
| Six-class goal detection F1 | 63.23% | 595 reused development images |
| Yellow localization precision / recall | 88.64% / 90.81% | Pi, 320×240 images, shape ignored |
| Context FP32 crop inference | 23.64 ms | Rebuilt ARMv6 runtime, bounded crop benchmark |
| Delivered paced replay | 7.93 FPS | Generated motion, not recorded flight |
| Live camera loop | 3.31 FPS | Unlabeled desk scene, not arena accuracy |

Small/orange proposals, balloon false positives, and independent arena testing
remain open. Later orientation smoke checks are not a repeat of the complete
benchmark. Read the [comparison protocol](docs/COMPETITION_COMPARISON.md) before
reusing numbers. No guaranteed range or 10 Hz control deadline is claimed.

## Repository map

| Path | Purpose |
| --- | --- |
| `src/dtr/` | Models, training, inference, proposals, tracking, viewer |
| `scripts/` | Data preparation, experiments, benchmarks, build/release tools |
| `configs/` | Tasks, reviewed training policies, frozen release configuration |
| `models/` | Small selected weights, metadata, provenance, verification manifests |
| `tests/` | Python regression and browser-policy unit tests |
| `docs/` | Guides, research evidence, history, publication checklist |
| `.github/` | CI and contribution templates |

Larger lineage weights and compiled runtime are release assets. Datasets,
recordings, caches, environments, and intermediate checkpoints stay outside Git.
The standalone pixel implementation is excluded; comparison runners require an
externally supplied baseline. Our own OpenCV proposals remain part of D-SHIFT.

## Development

```bash
uv run --locked --extra dev pytest -q
uv run --locked --extra dev ruff check .
node --test tests/frame-policy.test.cjs tests/goal-view.test.cjs
python3 scripts/verify_release.py
```

Keep v1 weights and source-bound implementation immutable; see [contributing](CONTRIBUTING.md).
The import/CLI remains `dtr`, and distribution name remains `dtr-mobilenetv4` for
compatibility. Neither names the student architecture.

## Security, attribution, and publication

The pinned research environments have **known dependency advisories**. Only load
trusted models and use local test inputs. Do not expose the viewer publicly.
See [SECURITY.md](SECURITY.md) and the [publication checklist](docs/PUBLIC_RELEASE.md).
The repo remains private until its owner explicitly authorizes publication.

Project code is [Apache-2.0](LICENSE). Upstream code, runtime dependencies, data,
and optional YOLO artifacts retain their own terms; see [third-party notices](THIRD_PARTY_NOTICES.md).
This is not an official MobileNet, TensorFlow, or university-endorsed implementation.
