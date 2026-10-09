"""Score the pixel-method balloon replay beside the neural pipeline's goldens on the same frames.

Development frames only; NOT flight qualification. Uses the development matching rule.
"""
import argparse
import json
from pathlib import Path

from dtr.data import read_json, sha256, write_json
from scripts.evaluate_red_blue_development import LABELS, detection_counts

PANELS = ("indoor", "original")


def panel_counts(frames, detections_per_frame):
    totals = {p: {label: dict(tp=0, fp=0, fn=0) for label in LABELS} for p in PANELS}
    for frame, detections in zip(frames, detections_per_frame, strict=True):
        counts = detection_counts(frame["truth"], detections)
        for label in LABELS:
            for key, value in counts[label].items():
                totals[frame["panel"]][label][key] += value
    return totals


def flat(totals):
    return " ".join(f"{p}:{label.split('_')[0]} {c['tp']}/{c['fp']}/{c['fn']}"
                    for p, by_label in totals.items() for label, c in by_label.items())


def review(base, golden, pixel_report, pi_replay):
    inputs = read_json(base / "inputs.json")
    if pixel_report["inputs_sha256"] != sha256(base / "inputs.json"):
        raise ValueError("Pixel report used different inputs")
    for row, frame in zip(pixel_report["frames"], inputs["frames"], strict=True):
        if row["path"] != frame["path"] or row["sha256"] != sha256(base / frame["path"]):
            raise ValueError(f"Frame mismatch: {frame['path']}")
    frames = inputs["frames"]
    pixel = [dict(threshold=run["threshold"], min_shape_size=run["min_shape_size"],
                  timing=run["timing"], color_timing=run["color_timing"],
                  counts=panel_counts(frames, [f["detections"] for f in run["frames"]]),
                  boxes_per_frame=sum(len(f["detections"]) for f in run["frames"]) / len(frames))
             for run in pixel_report["runs"]]
    model = {name: {mode: panel_counts(frames, [r["detections"] for r in rows])
                    for mode, rows in modes.items() if mode in ("mser", "mser_bright")}
             for name, modes in golden.items()}
    replay = {}
    for path in sorted(pi_replay.glob("*.json")):
        run = read_json(path)
        replay[path.stem] = dict(policy=run["policy"], model=run["model"], frames=run["frames"],
                                 mode_counts=run["mode_counts"],
                                 processing_ms=run["timing"]["processing_ms"],
                                 search_ms=run["timing"]["search_ms"]["mean_ms"],
                                 inference_ms=run["timing"]["inference_ms"]["mean_ms"])
    return dict(
        purpose="Pixel method vs neural pipeline on 16 development frames; NOT flight qualification",
        deployment_approved=False, flight_commands=None, test_evaluated=False,
        pixel_sha256=pixel_report["pixel"]["sha256"], colors=pixel_report["colors"],
        pixel_platform=pixel_report["platform"], pixel_health=[pixel_report["health_before"],
                                                              pixel_report["health_after"]],
        pixel=pixel, pixel_native_640x480_timing=pixel_report["native_640x480_timing"]["timing"]
        if pixel_report.get("native_640x480_timing") else None,
        model_counts=model, model_pi_replay=replay)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--pixel-report", required=True)
    parser.add_argument("--pi-replay", required=True, help="Scheduled-search Pi replay results")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    base = Path(args.base)
    result = review(base, read_json(base / "search-golden.json"), read_json(args.pixel_report),
                    Path(args.pi_replay))
    write_json(args.output, result)
    for run in result["pixel"]:
        print(f"pixel thr {run['threshold']:g}: {run['timing']['fps']:.2f} FPS  {flat(run['counts'])}")
    for name, modes in result["model_counts"].items():
        for mode, counts in modes.items():
            print(f"{name} {mode}: {flat(counts)}")
    for name, run in result["model_pi_replay"].items():
        print(f"{name}: {1000 / run['processing_ms']['mean_ms']:.2f} FPS")
    print(json.dumps(result["pixel_native_640x480_timing"]))


if __name__ == "__main__":
    main()
