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
- `network/`: laptop/Pi bench link; `firmware/`: ESP32 interface code.
- `configs/` and `models/`: configuration, frozen models and provenance.
- `lab_kit/`: portable laptop tools.

Datasets, recordings, credentials and intermediate weights are excluded. Keep
local services on trusted hosts; hardware output requires explicit configuration.

## Ground station

`network/` contains the Go laptop/Pi UDP link and embedded manual-control HTML,
JavaScript and CSS. On the flight laptop, join `DTR-Drone-Bench`, start the
packaged station, and open `http://127.0.0.1:8782`. The Pi hotspot, camera recorder
and recording bridge have separate startup services. Transfer-kit builders are
`scripts/package_ground_station.py` and `scripts/package_testflight_transfer.py`.
Generated kits and pairing credentials stay private and are not Git source.

## Systems integration handoff

The systems team may define the ESP32 packet format. The fields below describe
application data, not a requirement to use JSON, E1/E2/E3, or the laptop UDP wire
format. The existing ESP32 header is a proposal; the older reference sketch uses
a different format. The working ground link currently records commands on the
Pi; forwarding to the ESP32 and reading its application ACK remain integration
work.

| Input | Meaning |
| --- | --- |
| `forward`, `right`, `up`, `yaw` | Normalized manual intent in `[-1,1]`; positive means forward, right, up, and yaw right. These are not PWM values. |
| `hold` | True while input is held; release sends false with zero axes. Current browser samples at up to 10 Hz and the Pi expires held input after 300 ms. |
| Pause / Resume | Pause requests active position/altitude hold. Resume permits fresh commands; it must not replay a previous hold. |
| E-stop / reset | E-stop requests latched all-motor cutoff. Reset leaves Pause active and must not automatically arm or restart motors. |
| `intent` | Vision/mission state: `SEARCH`, `TRACK`, `LOST`, `FAN_ON`, `ENTER`, or `AVOID`. Repeated state updates are not repeated relative maneuvers. |
| `target_vector` | `yaw_deg`: relative turn, positive right; `elev_m`: height change, positive up; `travel_m`: forward travel after alignment. Degrees and metres, vehicle-relative, not arena coordinates. |
| Target / avoid observation | Label, confidence, box, geometry estimates and observation age/validity. Missing geometry is unavailable, not zero. |
| `fan_request` | Requested fan action; it does not confirm balloon capture. |

For today's receipt-only integration, the logical ACK needs the originating
session, sequence and message type, with `RECEIVED` or a rejection reason. Only
a complete validated message should be acknowledged. A bus-level I2C ACK alone
does not establish application receipt, and `RECEIVED` must not imply `APPLIED`.
No IMU or battery telemetry is required for this initial ACK step.

The current hardware assumption is Pi I2C bus 1, ESP32 address `0x42`, 100 kHz.
Systems must confirm the transport/pins, byte layout, units/scaling, integrity
check, ACK representation and command-expiry behavior so the Pi adapter can
match them. The ESP32 owns stabilization, motor mixing, arming, bounds and
physical failsafes. Launch and autonomous mission opcodes are not finalized;
the current console blocks Launch.

[Apache-2.0](LICENSE). Dependencies and data retain their own terms; see
[third-party notices](THIRD_PARTY_NOTICES.txt).
