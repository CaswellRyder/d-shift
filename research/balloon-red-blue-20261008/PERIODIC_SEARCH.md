# Periodic full-search experiment

2026-10-08. Research-only dispatch between the existing full two-pass MSER search
and the faster, non-equivalent bright-only policy. **Not a flight-ready tracker,
capture detector, controller or hard real-time guarantee.** Default deployment,
student weights, thresholds and training data remain unchanged.

## Fixed initial policy

The initial parameters are provisional engineering choices, not derived from a
measured blimp stopping distance or flight-control deadline. No parameter sweep
was performed against development labels.

- Full search on startup and after a failed search.
- At the next frame boundary, choose full search if at least **500 ms** has
  elapsed since the previous successful full search **started**.
- Otherwise allow at most **two completed bright-only frames** before a full
  search. A frame cap also prevents long full-search gaps at high frame rates.
- A full search already in progress consumes the time budget; the budget is not
  restarted at completion. A slow full search can therefore make the next frame
  a full search too, trading throughput for another complete observation.
- Dispatch is sequential. Beginning a second search before completing the first
  fails; invalid/backward clocks fail. Failed inference is propagated and resets
  the next search to full mode.

The scheduler does not preempt work. A frame that starts before the time limit
may run beyond it. `due_lateness_ns` records the actual excess at the next
dispatch; **500 ms is a dispatch threshold, not a maximum reacquisition time**.
Capture latency and object visibility can further delay a usable observation.

No observations or tracks are stored in the scheduler. Detections remain those
of the current frame. A target found only by a full search can disappear from
the bright-frame outputs; periodic rediscovery is not continuous target tracking.
No previous detection is relabeled fresh or sent to an ESP32.

## Generated recovery checks

The earlier counterexample is retained: a dark red disk on a brighter red field
has a valid full-pass proposal but no bright-only proposal. Repeat for blue.
At the first full frame the disk is absent; it appears at the first bright frame
and persists. Two bright searches miss it; the frame-cap-triggered full search
rediscovers its proposal. The test asserts candidate counts `[0, 0, 0, 1]` for
both color channels and both native-pointer modes.

This proves the fixture's proposal recovery and dispatch behavior. The timestamps
in that unit test are simulated, and no neural classification is performed.
It does not prove real-balloon recall, recovery of a transient target that leaves
before the full pass, a physical distance limit, or continuous-track safety.

Additional tests cover exact dispatch boundaries, slow frames/overruns, clock
types/order, pending-search state, failure recovery, forced-full controls,
timestamp-tampering rejection and inference exceptions.

Full current-working-tree suite: **721 passed, 2 skipped**; scoped Ruff checks
passed. Existing unrelated dirty edits were preserved and are not part of this
experiment's review or commit.

## Hardware experiment

Isolated bundle: `artifacts/balloon-pi-scheduled-search-20261008`, copied to
`/home/pacman/balloon-pi-scheduled-search-20261008`. Bundle SHA:
`ea21d1bf013b0a3467f722f27a04e893b4ec53ea05574d93a9d84cac507ca0aa`.
The builder preserves both existing policy goldens, all inputs and weights.
Only the research scheduler and its benchmark are added. The native reducer
compiled to the same binary as the preceding experiments.

The same wrapper runs always-full control and periodic intervention. Trial order:
full A, periodic A, periodic B, full B. Each trial warms both policies on all 16
images, then runs two timed rounds. The scheduler persists across the two rounds
but starts fresh after warmup. Each policy receives 64 timed samples across two
trials. Camera capture and disk/model loading are excluded. Golden comparisons
and file writes happen after the timed loop, so they do not alter dispatch.

The replay images are independent still-image workload entries in a fixed order,
**not a recorded motion sequence**. Its clock-driven policy mix measures compute
cost and dispatch intervals only. Count agreement against the appropriate
per-policy golden does not establish temporal recall. Original/indoor development
sources remain evaluation-only, including the entire reserved IMG family. No
final-test images were evaluated, calibrated on or admitted to training.

`review_scheduled_balloon_search.py` verifies bundle/report/log/model identities,
reconstructs each policy decision from recorded monotonic start/end timestamps,
checks frame order and per-policy predictions, and recomputes pooled timing. It
reports dispatch lateness without claiming a hard bound or flight qualification.
Both full-search start intervals and completed-result intervals are reported:
starting a full search is not the same as having its result available.
Full-search golden counts are compared against every timed frame, without
counting repeated images as independent accuracy samples.

## Completed paired Pi result

All four trials completed and their timestamp-replayed decisions passed review.
Every timed frame retains the full-search golden's TP/FP/FN counts on this small
panel. That is not exact prediction parity or evidence of temporal target recall.

- Always-full control: **3.1982 FPS**, mean **312.676 ms**, p95 **567.910 ms**.
- Periodic: **3.6008 FPS**, mean **277.717 ms**, p95 **571.767 ms**.
- Mean processing throughput improves **12.59%**; p95 does **not** improve.
- Periodic mix: 28 full and 36 bright searches across 64 timed samples.
  Neural calls fall from 5.625 to 4.8125 per frame.
- Full-search start intervals in periodic mode: mean **644.005 ms**, maximum
  **807.111 ms**. Maximum dispatch lateness beyond the 500-ms threshold is
  **307.111 ms**.
- Completed full-search result intervals: mean **645.753 ms**, p95
  **1058.769 ms**, maximum **1089.388 ms**. These are measured gaps between
  results, not an upper bound for future workloads or target reacquisition.
- All throttle flags remain `0x0`; no Python/camera workload remained at the
  final process check. No other vision process/profile ran during the trials.

The combination restores periodic access to proposals bright-only search can
omit, at a measured cost relative to bright-only processing. Its 3.60 FPS is
not directly interchangeable with the prior bright-only trial's 3.98 FPS or
camera-included rates. Capture time, scene order, policy mix and tail latency
matter. No deployment default or flight-readiness claim is changed.

Verified evidence: `periodic-search-pi-measurements.json`. Raw frame records
remain ignored under `artifacts/balloon-pi-scheduled-results-20261008`. The
reviewer recomputes timing from raw records and reports completed-result gaps
in addition to start intervals; it does not silently label the dispatch threshold
a deadline.

## Reproduction and next qualification work

Build a fresh bundle using `scripts.build_scheduled_balloon_pi_bundle` with the
verified bright-policy bundle as `--reference`. Compile the reducer on the Pi,
then run `pi_balloon_schedule_bench.py --policy full` or `--policy periodic`,
with the existing one-thread native-TFLite environment, `--rounds 2`, and new
output paths. Copy logs, then run:

```sh
.venv/bin/python -m scripts.review_scheduled_balloon_search --base artifacts/balloon-pi-scheduled-search-20261008 --results artifacts/balloon-pi-scheduled-results-20261008 --output artifacts/fresh-scheduled-summary.json
```

Before live integration, measure capture/sensor age as well as dispatch timing;
decide how missing observations age/expire downstream; test actual target
appearance, occlusion and loss on representative recorded sequences. The
500-ms/two-frame research policy does not itself establish an acceptable blimp
reacquisition deadline. No ESP32, serial, motor or network-configuration commands
are part of this experiment.

## Live camera harness (2026-10-09, paused)

`scripts/pi_balloon_live_schedule_bench.py` runs either policy on live OV5647
frames with the existing camera configuration (320x240 RGB888, 180-degree
transform, `queue=False`). The scheduler and sensor timestamps share
CLOCK_BOOTTIME. Besides each frame's own sensor-to-result age, it records the
**full-observation age**: at every result, the time since the newest full-search
frame's exposure. Bright frames can be fresh while this age grows. A full frame
with an invalid sensor timestamp makes the age unknown until the next valid full
search; an older observation is never relabeled fresh.
`scripts/review_live_scheduled_search.py` verifies hashes, replays scheduler
decisions from raw timestamps, recomputes every row-derived summary and pools
at least two trials per policy. Ten unit tests use a fake camera and clock.

Both scripts were copied beside the unchanged scheduled bundle on the Pi
(bundle, runtime and reducer hashes matched the replay trials). Planned order:
full A, periodic A, periodic B, full B, 60 frames each at a requested 10 FPS.

Only full A ran. Its frames were nearly black (mean RGB about 1/255; exposure
and gain saturated at 66.7 ms and 8.0), so the room lights were off. It recorded
5.00 FPS, about 155 ms search per frame, zero neural calls and a sensor-to-result
mean of 277 ms (max 293 ms), with all 60 timestamps valid. A dark scene does not
exercise proposals, so the comparison was stopped rather than completed. That
trial is excluded from any future pooled result; rerun all four in a lit scene
with new output names. Raw output stays ignored under
`artifacts/balloon-live-scheduled-results-20261009`.
