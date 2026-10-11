"""Capture bounded live frames at real sensor exposure and white-balance settings.

Auto exposure and white balance settle first. Each manual setting then scales the
settled exposure time, analogue gain or colour gains. No inference, no flight
commands; frames are unlabeled research inputs and are never training data.
"""

import argparse
import json
from pathlib import Path
import time

from PIL import Image

try:
    from pi_balloon_live_bench import capture_rgb, validate_camera_config
except ModuleNotFoundError:
    from scripts.pi_balloon_live_bench import capture_rgb, validate_camera_config

# name, exposure scale, gain scale, red colour-gain scale, blue colour-gain scale
SETTINGS = (
    ("manual_1.0", 1.0, 1.0, 1.0, 1.0),
    ("darker_0.6", 0.6, 1.0, 1.0, 1.0),
    ("darker_0.4", 0.4, 1.0, 1.0, 1.0),
    ("darker_0.25", 0.25, 1.0, 1.0, 1.0),
    ("brighter_gain_1.6", 1.0, 1.6, 1.0, 1.0),
    ("warm_wb", 1.0, 1.0, 1.25, 0.8),
    ("cool_wb", 1.0, 1.0, 0.8, 1.25),
)


def settle(camera, controls, limit=30):
    """Drop frames until the sensor reports the requested exposure, gain and colour gains."""
    for _ in range(limit):
        _, metadata = capture_rgb(camera)
        exposure, gain = metadata.get("ExposureTime", 0), metadata.get("AnalogueGain", 0)
        colour = metadata.get("ColourGains", (0, 0))
        if (abs(exposure - controls["ExposureTime"]) <= max(200, 0.05 * controls["ExposureTime"])
                and abs(gain - controls["AnalogueGain"]) <= 0.05 * controls["AnalogueGain"]
                and all(abs(a - b) <= 0.03 * b for a, b in zip(colour, controls["ColourGains"]))):
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frames", type=int, default=10)
    args = parser.parse_args()
    if not 1 <= args.frames <= 20:
        parser.error("Require 1..20 frames per setting")
    out = Path(args.output)
    out.mkdir(parents=False, exist_ok=False)

    from libcamera import Transform
    from picamera2 import Picamera2

    camera, rows = Picamera2(), []
    try:
        camera.configure(camera.create_video_configuration(
            main={"size": (320, 240), "format": "RGB888"},
            sensor={"output_size": (2592, 1944), "bit_depth": 10},
            controls={"FrameRate": 10},
            buffer_count=2,
            queue=False,
            transform=Transform(hflip=True, vflip=True),
        ))
        validate_camera_config(camera.camera_configuration())
        camera.start()
        time.sleep(3)

        def grab(name, settled):
            for i in range(args.frames):
                rgb, metadata = capture_rgb(camera)
                path = f"{name}-{i:02d}.png"
                Image.fromarray(rgb).save(out / path)
                rows.append(dict(path=path, setting=name, settled=settled,
                                 exposure_us=metadata.get("ExposureTime"),
                                 analogue_gain=metadata.get("AnalogueGain"),
                                 colour_gains=metadata.get("ColourGains"),
                                 lux=metadata.get("Lux")))

        grab("auto", True)
        base = rows[-1]
        exposure, gain, (red, blue) = base["exposure_us"], base["analogue_gain"], base["colour_gains"]
        for name, e, g, r, b in SETTINGS:
            controls = {"AeEnable": False, "AwbEnable": False, "ExposureTime": int(exposure * e),
                        "AnalogueGain": gain * g, "ColourGains": (red * r, blue * b)}
            camera.set_controls(controls)
            grab(name, settle(camera, controls))
        camera.set_controls({"AeEnable": True, "AwbEnable": True})
        time.sleep(3)
        grab("auto_after", True)
    finally:
        camera.close()
    report = dict(scope="Unlabeled live frames at real exposure settings; no inference or labels",
                  auto_exposure_us=exposure, auto_analogue_gain=gain, auto_colour_gains=[red, blue],
                  settings=[dict(zip(("name", "exposure", "gain", "red", "blue"), s)) for s in SETTINGS],
                  frames=rows, training_use=False, flight_commands=None)
    (out / "capture.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "frames"}, indent=2))
    print(sorted({(r["setting"], r["settled"], r["exposure_us"], round(r["analogue_gain"], 2)) for r in rows}))


if __name__ == "__main__":
    main()
