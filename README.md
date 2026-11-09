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
The laptop console stays available at `http://localhost:8782` or
`http://127.0.0.1:8782` while waiting for the hotspot. It reconnects on the
configured Wi-Fi interface when the Pi subnet appears; changing networks clears
old control input and requires a fresh Resume.

## Systems integration handoff

The systems team may define the ESP32 packet format. The fields below describe
application data, not a requirement to use JSON, E1/E2/E3, or the laptop UDP wire
format. The existing ESP32 header is a proposal; the older reference sketch uses
a different format. The ground link records commands on the Pi. An optional
manual-only ESP32 bridge now forwards held axes and reads E3 application acknowledgements. Vision
and autonomous forwarding remain unfinished.

| Input | Meaning |
| --- | --- |
| `forward`, `right`, `up`, `yaw` | Normalized manual intent in `[-1,1]`; positive means forward, right, up, and yaw right. These are not PWM values. |
| `hold` | True while input is held; release sends false with zero axes. Current browser samples at up to 10 Hz and the Pi expires held input after 300 ms. |
| Pause / Resume | The manual bench controller stops propulsion and disarms on Pause. Resume permits fresh commands and requires neutral arming again; it does not restore a hold. |
| E-stop / reset | E-stop requests latched all-motor cutoff. Reset leaves Pause active and must not automatically arm or restart motors. |
| `intent` | Vision/mission state: `SEARCH`, `TRACK`, `LOST`, `FAN_ON`, `ENTER`, or `AVOID`. Repeated state updates are not repeated relative maneuvers. |
| `target_vector` | `yaw_deg`: relative turn, positive right; `elev_m`: height change, positive up; `travel_m`: forward travel after alignment. Degrees and metres, vehicle-relative, not arena coordinates. |
| Target / avoid observation | Label, confidence, box, geometry estimates and observation age/validity. Missing geometry is unavailable, not zero. |
| `fan_request` | Requested fan action; it does not confirm balloon capture. |

The manual bridge uses E1/E3 packets with session, sequence, message type and
CRC. `RECEIVED` confirms validation; `APPLIED` reports the ESP32's software
control state. Neither measures ESC response, RPM or physical movement. A Pi
receipt confirms Pi handling only; inspect Controller telemetry for ESP status.

The current hardware assumption is Pi I2C bus 1, ESP32 address `0x42`, 100 kHz.
The manual wire layout is implemented in the Pi adapter and ESP32 sketch.
Systems must physically verify wiring, ESC signal compatibility, output polarity,
motor mapping and cutoff behavior. This bench firmware owns arming, output
bounds and command expiry; stabilization and flight control are unfinished.
Launch and autonomous mission opcodes are not finalized; the current console
blocks Launch.

## Manual motor bench

Firmware is `firmware/esp32_manual_motors/esp32_manual_motors.ino`, targeting
Seeed XIAO ESP32-S3 with Arduino-ESP32 3.3.12 and USB CDC enabled. Start the Pi
link with its usual flags plus `--motor-i2c-bus 1`; motor output is disabled by
default. Other operating systems can build/test the link but cannot open I2C.

With propellers removed, verified wiring and an independent power disconnect:
select Resume and Enable controls, then close the physical D9 enable switch.
Hold **Space** or **Hold to arm / deadman** with no direction for two seconds.
Keep that control held while pressing W/A/S/D, R (up), F (down), T (auxiliary),
or a direction button. Releasing a direction returns to neutral while armed;
releasing the deadman, losing input, Pause or E-stop stops and disarms. A new
neutral hold is required. A Pi process restart also requires opening and closing
the physical enable switch. Pause provides no position or altitude hold.

The provisional mapping uses UP for both thrust outputs, RIGHT for differential
thrust, FORWARD for a dedicated motor and YAW for the auxiliary. RIGHT alone
can run one thrust output. Reverse/descent produce no thrust. All four outputs
are capped at 1000–1250 microseconds, 50 Hz. Confirm the actual ESC protocol and
motor geometry before powering them. Full wiring, failure behavior and the
remaining hardware checks are in
[the bench integration guide](firmware/esp32_manual_motors/BENCH_INTEGRATION.txt).

Software checks:

```sh
(cd network && go test -race ./... && go vet ./...)
node --test network/laptop/web/*.test.mjs
scripts/test_manual_motors.sh
arduino-cli compile --fqbn esp32:esp32:XIAO_ESP32S3 firmware/esp32_manual_motors
```

[Apache-2.0](LICENSE). Dependencies and data retain their own terms; see
[third-party notices](THIRD_PARTY_NOTICES.txt).
