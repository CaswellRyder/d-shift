"""Read-only development replay of goal lab takes (never training or qualification).

Each supplied take is expected to contain one yellow triangle. Scene labels are
deliberately ignored. JSON retains every detected frame and unexpected detection.
"""

import argparse
from collections import Counter
import json
from pathlib import Path

import cv2

from dtr.goal_square import DEFAULT_MAX_CROPS, find_goals
from dtr.runtime import Predictor


def replay(takes, predictor):
    results = []
    for take in takes:
        paths = sorted((take / "raw").glob("*.png"))
        if not paths:
            raise ValueError(f"No raw PNG frames: {take}")
        hits, unexpected, crops = [], [], Counter()
        for path in paths:
            if path.name.startswith("IMG_"):
                raise ValueError("Reserved IMG_ family is excluded")
            rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
            found = find_goals(rgb, predictor)
            crops[found["neural_calls"]] += 1
            if any(d["label"] == "yellow_triangle" for d in found["detections"]):
                hits.append(path.name)
            unexpected.extend(dict(frame=path.name, **d) for d in found["detections"]
                              if d["label"] != "yellow_triangle")
        row = dict(take=take.name, frames=len(paths), yellow_triangle_frames=len(hits),
                   detected_frames=hits, unexpected=unexpected, crop_histogram=dict(crops))
        results.append(row)
        print(f"{take.name}: {len(hits)}/{len(paths)} yellow_triangle; "
              f"{len(unexpected)} other detections; crops {dict(crops)}", flush=True)
        for detection in unexpected:
            print(json.dumps(detection), flush=True)
    return dict(scope="development only; scene labels ignored; one yellow triangle per take",
                max_crops=DEFAULT_MAX_CROPS, threshold=.8, color_threshold=.8, takes=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("takes", type=Path, nargs="+")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    predictor = Predictor(args.model, allow_unvalidated=True)
    result = replay(args.takes, predictor)
    result["model_sha256"] = predictor.metadata["sha256"]
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
