"""Count accepted detections on live Pi frames captured with no balloons in view.

Every accepted, unsuppressed red/blue detection is a false positive. Frames come
from `pi_capture_exposure_frames.py`; the search is the bundle's exact MSER
search, which the Pi's native direct search reproduces on the bundle goldens.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, write_json
from dtr.runtime import Predictor
from scripts.pi_balloon_search_bench import process_frame


def evaluate(captures, models):
    frames = []
    for capture in captures:
        capture = Path(capture)
        report = read_json(capture / "capture.json")
        frames += [(capture.name, row, np.asarray(Image.open(capture / row["path"]).convert("RGB")))
                   for row in report["frames"]]
    results = {}
    for name, path in models.items():
        predictor = Predictor(path, allow_unvalidated=True)
        by_setting, events = {}, []
        for capture, row, rgb in frames:
            accepted = [d for d in process_frame(rgb, predictor, "mser")["detections"]
                        if d["accepted"] and not d.get("suppressed")]
            counts = by_setting.setdefault(row["setting"], dict(frames=0, red_fp=0, blue_fp=0,
                                                                frames_with_fp=0))
            counts["frames"] += 1
            counts["frames_with_fp"] += bool(accepted)
            for d in accepted:
                counts["red_fp" if d["label"] == "red_balloon" else "blue_fp"] += 1
                events.append(dict(capture=capture, path=row["path"], label=d["label"],
                                   box=d["box"], score=max(d["scores"])))
        results[name] = dict(model=str(path), model_sha256=sha256(path), by_setting=by_setting,
                             red_fp=sum(c["red_fp"] for c in by_setting.values()),
                             blue_fp=sum(c["blue_fp"] for c in by_setting.values()),
                             events=events)
    return dict(scope="Live Pi negatives at real exposure settings; false positives only, no recall",
                captures={Path(c).name: sha256(Path(c) / "capture.json") for c in captures},
                frames=len(frames), models=results, script_sha256=sha256(__file__))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", action="append", required=True)
    parser.add_argument("--model", action="append", required=True, help="alias=path.tflite")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    result = evaluate(args.capture, dict(m.split("=", 1) for m in args.model))
    write_json(args.output, result)
    for name, r in result["models"].items():
        print(name.ljust(16), "red", r["red_fp"], "blue", r["blue_fp"],
              {k: (v["red_fp"], v["blue_fp"]) for k, v in r["by_setting"].items()})
