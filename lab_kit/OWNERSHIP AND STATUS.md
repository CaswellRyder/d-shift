# Ownership and status

This separates implemented software from calibrated behavior and team-owned flight control.

| Owner | Original section | What is implemented here | What remains |
|---|---|---|---|
| Ryder / R | Find_Target balloon and goal | Detection/tracking interface; camera and body coordinate estimates; lens/mount configuration; target reports; lab console; captures, diagnostics and logs | Measured lens/size/mount calibration, lit moving-target trials, measured XYZ validation and acceptance |
| Connor / C | Vector_Creation / navigation | Draft interface support only: geometric yaw/climb pointing errors and proposed current XYZ offset fields for review | Navigation/control algorithm, IMU integration and clock alignment, tuned gains, approach distances, actuator mixing and response validation |
| Josh / J | Send_to_Arduino / Pi-to-ESP32 transport | Draft JSON serialization, exact outgoing log files and message preview for review | Agreed wire contract, actual UART/radio transport, firmware parser, acknowledgements, rate/baud/framing choices, watchdog and link-loss testing |
| ESP32/mechanical team | Fast IMU/actuator loop, physical safety and possession sensors | No firmware or actuator implementation | Servo/PWM limits and calibration, flight controller, sonar integration, independent emergency stop, reversible thrust/floor-stop decisions |

## Important distinction

We have **not completed Connor's or Josh's functional sections**. We have, however, touched their interface boundaries. Calling them wholly untouched would be inaccurate.

The Find_Target implementation and test tooling are built. Calling Ryder's entire section complete would also be too strong until the measured calibration and live lab checks pass.

## Shared source changes to disclose

- `src/dtr/flight.py`: records the selected track ID so logged coordinates match the exact target chosen by the existing mission. No new navigation controller or shooting state was added.
- `src/dtr/flight_pipeline.py`: shares the existing frame processing order between runner and lab recorder, with phase-epoch filtering and optional coordinate telemetry.
- `scripts/pi_flight_loop.py`: calls that shared step, preserves camera sensor metadata and saves draft coordinate logs alongside the existing v1 lines.
- `src/dtr/flight_telemetry.py`: estimates relative coordinates, computes geometric pointing errors, labels missing inputs/calibration, and serializes a draft handoff. This is interface prototyping around Connor/Josh's boundaries, not an approved controller/transport implementation.
- `scripts/open_flight_lab.py` and the portable `Laptop/` launcher: laptop-to-Pi lab access over SSH. This is separate from Josh's Pi-to-ESP32 flight link.

## Items still absent from the proposed master loop

No complete autonomous launch/capture/shot/verify/reset state machine, wireless LAUNCH/DESCEND/KILL application, physical emergency-stop verification, full vehicle simulator, ESP32 firmware, tuned steering, or flight acceptance has been delivered. The legacy `FAN_ON`/`ENTER` messages in logs do not prove the later SHOOT/VERIFY/post-score design exists.

No serial or motor commands were issued. No measured calibration or physical flight-readiness claim is made by this package.
