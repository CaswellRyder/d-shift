# Pi Zero W research handoff

Live-checked 2026-10-05: Raspberry Pi Zero W Rev 1.1, ARMv6 (`armv6l`), Raspbian 13,
Python 3.13.5. OV5647 detection and actual 320x240 camera capture passed. USB SSH,
Wi-Fi, DNS and HTTPS work. No flight actuation, boot/network changes or package replacement.
The isolated [two-method comparison bundle](PI_COMPARISON.md) is installed under
`/home/pacman/pi-comparison-20261005`; that frozen bundle remains unchanged.
On 2026-10-06, the current standalone pixel camera capture was updated for the
upside-down mount; see [camera orientation](CAMERA_ORIENTATION.md).

The newer [optimization experiment](PI_OPTIMIZATION.md) adds a longer-trained goal
student, measured FP32 native inference, and bounded temporal tracking. It lives
separately under `/home/pacman/pi-optimization-20261005`; it does not replace this
frozen INT8 comparison or authorize flight.

## Minimal runtime

The `dtr.runtime` package needs Python 3, NumPy, Pillow and an inference backend.
It tries `tflite_runtime`, then TensorFlow on training hosts, then the small `native_tflite`
C API adapter. The Pi already has `libtensorflow-lite2.20.0`; its native C API ran both
real-data INT8 students and passed 30 golden label/score comparisons with the Mac.
No modern ARMv7/ARM64 wheel or Python TensorFlow installation was needed.

From the comparison bundle on the Pi:

```python
from dtr.runtime import Predictor, benchmark

predictor = Predictor('models/goal.tflite', allow_unvalidated=True)
print(predictor.predict('test_crop.png'))
print(benchmark('models/goal.tflite', 'test_crop.png', count=100, allow_unvalidated=True))
```

This is a local bench test, not approval for flight. Never mark a synthetic model as approved.
Input is RGB uint8, not OpenCV's default BGR. The runtime applies bilinear crop resizing,
quantization scale/zero point, integer inference, output dequantization and softmax.

Remaining integration gates (initial camera capture and native crop inference passed):

1. Repeat camera trials under arena lighting with measured frame age and labeled recordings.
2. Golden-crop verification must remain reproducible after any model/runtime changes.
3. p50/p95 inference and full camera-to-observation latency measured on the actual Pi.
4. Proposal recall and wrong-color false positives tested on held-out arena recordings.
5. Camera loop drops stale frames, limits candidate count, and has target-loss handling.
6. Separate microcontroller watchdog owns motor safety; perception does not prove capture or score.

No Mac timing is a Pi FPS claim. A bounding box around a goal is not its traversable opening.
