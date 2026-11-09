# D-SHIFT

Computer vision, target tracking and experimental mission interfaces for Defend
the Republic blimps using a Raspberry Pi Zero W.

Research software; calibration, controller integration and flight qualification
remain unfinished. Included v1 balloon models detect green/purple, not red/blue.

## Desktop setup

Use Python 3.11 and [uv](https://docs.astral.sh/uv/). No Pi or camera is needed:

```sh
uv sync --locked --extra dev
python3 scripts/verify_current_models.py
uv run --locked python scripts/verify_current_models.py --smoke
```

The Pi uses a separate ARMv6 runtime; do not install desktop TensorFlow on it.

## Code

- `src/dtr/`: perception, tracking, geometry and mission interfaces.
- `scripts/`: launchers, calibration, training and evaluation tools.
- `configs/` and `models/`: configuration, frozen models and provenance.
- `lab_kit/`: portable laptop tools.

Datasets, recordings, credentials and intermediate weights are excluded. Keep
local services on trusted hosts; hardware output requires explicit configuration.

[Apache-2.0](LICENSE). Dependencies and data retain their own terms; see
[third-party notices](THIRD_PARTY_NOTICES.txt).
