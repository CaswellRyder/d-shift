# Find_Target lab console and laptop kit

Download the **Find_Target lab kit** prerelease ZIP from this repository's [Releases](https://github.com/CaswellRyder/d-shift/releases), then copy and extract the entire folder on your Mac laptop. The repository's `lab_kit` directory contains launcher source; the release ZIP adds the verified Pi models/runtime and generated manifests needed to run it.

## On the laptop

1. Connect the existing configured Pi Zero W through its USB data port and allow it to boot.
2. Double-click `Verify Package.command`, then `Start Flight Lab.command` in the extracted folder.
3. Enter the `pacman` SSH password in Terminal when SSH asks. Verify the Pi's host fingerprint on first connection through a trusted source.
4. The browser opens `http://127.0.0.1:8770`. Choose a target, enter measured lens-to-target distance and press **Capture 5 seconds**.

The laptop needs macOS, Python 3.9+, SSH/SCP and a browser. It needs no repository checkout, pip packages, TensorFlow or OpenCV. The Pi runs the camera, models, tracker and coordinate estimates. `connection.json` is editable and excluded from immutable package checksums; `connection.example.json` preserves the supplied defaults.

The launcher verifies a known running Pi console before reusing it. Otherwise it installs the self-contained archive in a new isolated directory. It does not stop other Pi processes, overwrite changed installations, change networking or install operating-system packages. A capture already in progress must finish before reconnecting or downloading.

This package targets the existing ARMv6 Pi Zero W with Python, NumPy, Pillow, OpenCV, Picamera2 and libcamera already installed. It is not a fresh Raspberry Pi OS image. The included native inference library is for the Pi, not macOS.

## Recording and coordinates

The console uses the same frame-processing step as `scripts/pi_flight_loop.py`. During a five-second take it buffers RGB frames and live results in RAM, then creates PNGs, masks, selected-box overlays, timestamped replay, an AVI when supported, JSONL logs and a downloadable ZIP. Preview and artifact transfers pause/conflict with the timed capture. Actual frame counts, gaps and copy overhead are recorded; 20 Hz is the requested camera rate, not an acceptance guarantee.

Each take includes `raw/`, `masks/`, `tracker-masks/`, `overlay/`, `frames.jsonl`, `wire-v1.jsonl`, `esp32-draft.jsonl`, `mapping.json`, `scene.json`, `summary.json` and `download.zip`. Masks are reconstructed afterward using the same flight functions; model inference and coordinate selection are not rerun to manufacture results. AVI playback uses measured average frame rate; timestamped images retain per-frame timing.

The proposed handoff is **X right, Y up, Z forward**, in metres. A centred target 3 m ahead is `(0, 0, 3)` with an aligned camera. It is the current target offset, not the change between frames or the drone's arena position. Camera and vehicle origins differ when the camera/funnel mounting is offset. Tape distance is a separate reference label and never an input to coordinate estimation. A target 5 m forward and 3 m above the camera has a diagonal range of about 5.83 m.

The lens currently uses an ideal equidistant estimate. Lens calibration, balloon size and mounting measurements remain outstanding; goal pose can bias range. The UI and draft packet mark estimates and missing inputs. IMU, world heading, forward thrust and servo/PWM outputs are unavailable, `control_ready` is false, and nothing is transmitted to an ESP32.

Use `Download Captures.command` to copy the completed take ZIPs listed by the viewer into `Captures/`. The history API currently exposes the latest 100 takes; files remain on the Pi. Interrupted downloads and recordings retain explicit incomplete status.

## Ownership and status

Ryder's Find_Target implementation and lab tools are built; measured calibration and live acceptance remain pending. Connor's navigation/controller and Josh's real Pi-to-ESP32 transport are not implemented. Draft pointing math, packet formats, shared-runner instrumentation and logs do touch their interface boundaries. See `OWNERSHIP AND STATUS.md` for the exact distinction.

The full SHOOT/VERIFY/post-score flow, tuned controller, UART/firmware integration and hardware safety validation are not completed. Synthetic software replay is not a flight dynamics simulator. Models and runtime remain research artifacts, not approved for flight. Source test and browser/Pi smoke evidence do not establish measured target accuracy or flight readiness.

## Rebuild a release locally

```sh
python3 scripts/package_flight_lab.py \
  --balloon-model /path/to/green-purple-context.tflite \
  --goal-model /path/to/student.float.tflite \
  --runtime /path/to/libtensorflowlite_c.candidate.so \
  --source-revision COMMIT_SHA \
  --output artifacts/Find-Target-Lab-Kit
```

Each model requires its original adjacent `.json` metadata. The builder verifies hashes, preserves metadata, and pins the native runtime against the frozen build evidence. It generates the Pi archive, integrity manifest and transfer ZIP. Weights/runtime are release assets, not new Git objects. The allowlist excludes datasets, recordings, conversation archives, class documents, credentials and other generated local material.

To fetch the prerelease from a laptop with GitHub CLI:

```sh
gh release download lab-kit-2026-11-02 --repo CaswellRyder/d-shift \
  --pattern 'find-target-lab-kit-2026-11-02.zip' --dir ~/Downloads
```

Or download the ZIP directly from the repository's Releases page. Hugging Face is not required for this kit.

## Checks

```sh
python3 -B -m unittest discover -s lab_kit/Laptop/tests -v
.venv/bin/python -m pytest -q
```

The laptop launcher tests are dependency-free and mock network/Pi actions. Actual first connection from the intended laptop remains a hardware check.
