"""Score scenes recorded with the Pi browser viewer.

Labels are the balloon counts typed before each scene, so scoring is per-frame
count agreement, not box accuracy: a frame is exact when its accepted red and
blue counts equal the counts in view. With --base, every saved frame is also
replayed on this machine, first with the Pi's own selection (a parity check
against the recorded detections), then with the research selection rules and
with stride-2 pixel search (the fast path used between full searches).
Lab frames are evaluation data only; nothing here trains or tunes a model.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, write_json

try:
    from pi_balloon_search_bench import SELECTIONS, check_predictions, process_frame
    from research_pixel_proposals import CvPixelMasks, process_frame as pixel_frame
except ModuleNotFoundError:
    from scripts.pi_balloon_search_bench import SELECTIONS, check_predictions, process_frame
    from scripts.research_pixel_proposals import CvPixelMasks, process_frame as pixel_frame

LABELS = ("red_balloon", "blue_balloon")
# Same fixed pixel settings as scripts/compare_pi_search_paths.py.
PIXEL = "pixel2"
PIXEL_COLORS, PIXEL_THRESHOLD, PIXEL_MIN_AREA = ((178, 28, 48), (26, 86, 179)), 75, 53
SEARCHES = SELECTIONS + (PIXEL,)


def load_scene(folder):
    folder = Path(folder)
    meta = read_json(folder / "scene.json")
    rows = [json.loads(line) for line in (folder / "frames.jsonl").read_text().splitlines() if line]
    return meta, rows


def counts(detections):
    return {label: sum(1 for d in detections if d["accepted"] and d["label"] == label) for label in LABELS}


def agreement(frames, expected):
    """frames: one detection list per frame; expected: label -> count in view."""
    result = dict(frames=len(frames), exact_frames=0)
    for label in LABELS:
        result[label] = dict(in_view=expected[label], extra_frames=0, missing_frames=0,
                             extra=0, missing=0)
    for detections in frames:
        found = counts(detections)
        result["exact_frames"] += all(found[k] == expected[k] for k in LABELS)
        for label in LABELS:
            gap = found[label] - expected[label]
            row = result[label]
            row["extra_frames"] += gap > 0
            row["missing_frames"] += gap < 0
            row["extra"] += max(gap, 0)
            row["missing"] += max(-gap, 0)
    result["exact_rate"] = result["exact_frames"] / len(frames) if frames else None
    return result


def percentiles(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return dict(p50=float(np.median(values)), p95=float(np.percentile(values, 95)), max=float(max(values)))


def timing(rows):
    stamps = [r["sensor_timestamp_ns"] for r in rows if r.get("timestamp_valid")]
    gaps = np.diff(stamps) / 1e9 if len(stamps) > 1 else []
    return dict(
        fps=float(1 / np.mean(gaps)) if len(gaps) and np.mean(gaps) > 0 else None,
        processing_ms=percentiles([r.get("processing_ms") for r in rows]),
        sensor_to_result_ms=percentiles([r.get("sensor_to_result_ms") for r in rows]),
        full_search_fraction=(sum(r.get("mode") == "mser_direct" for r in rows) / len(rows)) if rows else None,
        policies=sorted({r.get("policy") for r in rows if r.get("policy")}),
    )


class Replay:
    """Desktop replay of saved frames with the bundle's model and native search."""

    def __init__(self, base, region_library):
        from dtr.native_balloon_regions import NativeBalloonRegions
        from dtr.runtime import Predictor

        self.root = Path(base).resolve()
        self.inputs = read_json(self.root / "inputs.json")
        self.native = NativeBalloonRegions(region_library, direct=True)
        self.predictor_type, self.predictors = Predictor, {}
        self.masks = CvPixelMasks(PIXEL_COLORS, PIXEL_THRESHOLD)

    def predictor(self, alias, digest):
        if alias not in self.predictors:
            model = self.root / self.inputs["models"][alias]["path"]
            if sha256(model) != digest:
                raise ValueError(f"{alias} in {self.root} is not the model the scene was recorded with")
            self.predictors[alias] = self.predictor_type(model, allow_unvalidated=True)
        return self.predictors[alias]

    def scene(self, folder, meta, rows, selections):
        predictor = self.predictor(meta["model"], meta["model_sha256"])
        replayed = {s: [] for s in selections}
        parity = dict(frames=0, mismatched_frames=0, max_score_delta=0.0)
        for row in rows:
            rgb = np.asarray(Image.open(Path(folder) / row["path"]).convert("RGB"))
            for selection in selections:
                if selection == PIXEL:
                    detections = pixel_frame(rgb, predictor, self.masks, PIXEL_MIN_AREA, stride=2)["detections"]
                else:
                    detections = process_frame(rgb, predictor, row["mode"], self.native, selection)["detections"]
                replayed[selection].append(detections)
                if selection == "parts":
                    parity["frames"] += 1
                    try:
                        delta = check_predictions(detections, row["detections"])
                        parity["max_score_delta"] = max(parity["max_score_delta"], delta)
                    except ValueError:
                        parity["mismatched_frames"] += 1
        return replayed, parity


def score(folders, replay=None, selections=SEARCHES):
    scenes = []
    for folder in folders:
        meta, rows = load_scene(folder)
        if meta.get("training_use") is not False:
            raise ValueError(f"{folder} is not marked as evaluation-only")
        expected = {"red_balloon": meta["red_in_view"], "blue_balloon": meta["blue_in_view"]}
        entry = dict(folder=str(folder), scene=meta["scene"], take=meta["take"], notes=meta["notes"],
                     model=meta["model"], source=meta.get("source"), in_view=expected,
                     pi=agreement([r["detections"] for r in rows], expected), timing=timing(rows))
        if replay is not None and rows:
            replayed, parity = replay.scene(folder, meta, rows, selections)
            entry["desktop_parity"] = parity
            entry["replayed"] = {s: agreement(frames, expected) for s, frames in replayed.items()}
        scenes.append(entry)
    return scenes


def find_scenes(paths):
    found = set()
    for path in map(Path, paths):
        if (path / "scene.json").exists():
            found.add(path)
        found.update(p.parent for p in path.rglob("scene.json"))
    return sorted(found)


def table(scenes):
    lines = [f"{'scene':28} {'in view':>8} {'frames':>6} {'exact':>6} {'red +/-':>9} {'blue +/-':>9} "
             f"{'fps':>5} {'p95 ms':>7}"]

    def line(name, view, a, t=None):
        red, blue = a["red_balloon"], a["blue_balloon"]
        rate = f"{a['exact_rate']:.0%}" if a["exact_rate"] is not None else "-"
        fps = f"{t['fps']:.1f}" if t and t["fps"] else ""
        p95 = f"{t['sensor_to_result_ms']['p95']:.0f}" if t and t["sensor_to_result_ms"] else ""
        return (f"{name:28} {view:>8} {a['frames']:>6} {rate:>6} "
                f"{'+%d/-%d' % (red['extra'], red['missing']):>9} "
                f"{'+%d/-%d' % (blue['extra'], blue['missing']):>9} {fps:>5} {p95:>7}")

    for s in scenes:
        name = f"{s['scene']}-{s['take']:02d}"
        view = f"{s['in_view']['red_balloon']}r {s['in_view']['blue_balloon']}b"
        lines.append(line(name, view, s["pi"], s["timing"]))
        for selection, a in s.get("replayed", {}).items():
            lines.append(line(f"  desktop {selection}", "", a))
        if "desktop_parity" in s:
            p = s["desktop_parity"]
            lines.append(f"  parity with Pi: {p['frames'] - p['mismatched_frames']}/{p['frames']} frames, "
                         f"max score delta {p['max_score_delta']:.1e}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sessions", nargs="+", help="scene folders, or folders that contain them")
    parser.add_argument("--output", help="write the full results as JSON")
    parser.add_argument("--base", help="Pi bundle directory; enables desktop replay")
    parser.add_argument("--region-library", help="native region library for this machine")
    parser.add_argument("--selections", nargs="+", choices=SEARCHES, default=list(SEARCHES))
    args = parser.parse_args()
    if bool(args.base) != bool(args.region_library):
        parser.error("--base and --region-library go together")
    folders = find_scenes(args.sessions)
    if not folders:
        parser.error("No scene.json found")
    replay = Replay(args.base, args.region_library) if args.base else None
    selections = ["parts"] + [s for s in args.selections if s != "parts"]
    scenes = score(folders, replay, selections)
    print(table(scenes))
    if args.output:
        write_json(args.output, dict(
            scope="Bench scenes from the Pi viewer; labels are typed counts, not reviewed boxes; "
                  "evaluation only, not training or tuning data",
            scenes=scenes))


if __name__ == "__main__":
    main()
