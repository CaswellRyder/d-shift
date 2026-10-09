"""Synthetic pans over still development photos: scheduler reacquisition, not real motion.

Each admitted EngDes2 development photo (IMG_ family, evaluation-only) is viewed
through a deterministic moving square window on its 640x640 export, so labeled
targets leave and re-enter view. Both search modes are scored on every 10 FPS
view, then the full and periodic policies are replayed through SearchSchedule
with a fixed camera/latency model taken from earlier Pi logs. No blur, lighting
change, balloon motion, occlusion or content-dependent cost is modeled.
"""

import argparse
import io
import json
import math
from pathlib import Path
import zipfile

import cv2
import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, write_json
from dtr.native_balloon_regions import NativeBalloonRegions
from dtr.research_search_schedule import SearchSchedule
from dtr.runtime import Predictor
from dtr.tracking import iou
from scripts.evaluate_red_blue_development import LABELS
from scripts.evaluate_reviewed_balloon_scenes import reviewed_frames
from scripts.pi_balloon_search_bench import check_predictions, process_frame
from scripts.pi_red_blue_bench import statistics, verify_bundle

SOURCE = 640
OUTPUT = (320, 240)
MODES = ("mser_direct", "mser_bright")
PERIOD_NS = 100_000_000
SECONDS = 12
PRESENT_FRACTION = 0.75
MS = 1_000_000


def window(t, phase):
    """Square source window (x0, y0, side); 320..416 px, so output never upsamples."""
    side = 368 + 48 * math.sin(2 * math.pi * t / 7.0 + phase)
    room = (SOURCE - side) / 2
    cx = SOURCE / 2 + room * math.sin(2 * math.pi * t / 4.0 + 1.7 * phase)
    cy = SOURCE / 2 + room * math.sin(2 * math.pi * t / 5.5 + 2.3 * phase)
    s = int(round(side))
    x0 = min(max(int(round(cx - side / 2)), 0), SOURCE - s)
    y0 = min(max(int(round(cy - side / 2)), 0), SOURCE - s)
    return x0, y0, s


def render(source, win):
    x0, y0, s = win
    return cv2.resize(source[y0 : y0 + s, x0 : x0 + s], OUTPUT, interpolation=cv2.INTER_AREA)


def view_truth(targets, win):
    """Clip source boxes to the window; partly visible targets are ignore regions."""
    x0, y0, s = win
    sx, sy = OUTPUT[0] / s, OUTPUT[1] / s
    rows = []
    for index, target in enumerate(targets):
        a, b, c, d = target["source_box"]
        ca, cb, cc, cd = max(a, x0), max(b, y0), min(c, x0 + s), min(d, y0 + s)
        visible = max(0, cc - ca) * max(0, cd - cb) / ((c - a) * (d - b))
        if visible == 0:
            continue
        box = [(ca - x0) * sx, (cb - y0) * sy, (cc - x0) * sx, (cd - y0) * sy]
        present = visible >= PRESENT_FRACTION and min(box[2] - box[0], box[3] - box[1]) >= 4
        rows.append(
            dict(target=index, label=target["label"], box=box, visible=visible, present=present)
        )
    return rows


def inside(box, region):
    area = (box[2] - box[0]) * (box[3] - box[1])
    overlap = max(0, min(box[2], region[2]) - max(box[0], region[0])) * max(
        0, min(box[3], region[3]) - max(box[1], region[1])
    )
    return overlap / area if area > 0 else 0.0


def score_view(truth, detections):
    """Confidence-ordered one-to-one matching at IoU 0.5 against present targets.

    Unmatched accepted boxes mostly inside a partly visible same-color target are
    ignored rather than counted as false positives.
    """
    matched, fp, ignored = set(), 0, 0
    for detection in sorted(detections, key=lambda r: -r["score"]):
        if not detection["accepted"] or detection["label"] not in LABELS:
            continue
        options = [
            (iou(t["box"], detection["box"]), t["target"])
            for t in truth
            if t["present"] and t["target"] not in matched and t["label"] == detection["label"]
        ]
        overlap, target = max(options, default=(0.0, -1))
        if overlap >= 0.5:
            matched.add(target)
        elif any(
            not t["present"]
            and t["label"] == detection["label"]
            and inside(detection["box"], t["box"]) >= 0.5
            for t in truth
        ):
            ignored += 1
        else:
            fp += 1
    present = [t["target"] for t in truth if t["present"]]
    return dict(
        hits=sorted(matched),
        missed=sorted(t for t in present if t not in matched),
        fp=fp,
        ignored=ignored,
    )


def simulate(policy, scored, cost_ns, delivery_ns, period_ns=PERIOD_NS):
    """Replay a policy over per-tick scores with a fixed camera and cost model.

    Frame k is exposed at k*period and delivered delivery_ns later. Like
    queue=False capture, each search takes the first frame delivered at or after
    the previous result; nothing preempts a running search.
    """
    scheduler, events, result = SearchSchedule(), [], 0
    while True:
        tick = max(0, -(-(result - delivery_ns) // period_ns))
        if tick >= len(scored):
            return events
        start = tick * period_ns + delivery_ns
        decision = scheduler.begin(start, force_full=policy == "full")
        result = start + cost_ns[decision["mode"]]
        scheduler.finish(result)
        events.append(
            dict(
                tick=tick,
                mode=decision["mode"],
                reason=decision["reason"],
                sensor_ns=tick * period_ns,
                result_ns=result,
                **scored[tick][decision["mode"]],
            )
        )


def episodes(present):
    """Maximal [start, end) tick runs where a target is present."""
    runs, start = [], None
    for index, flag in enumerate([*present, False]):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            runs.append((start, index))
            start = None
    return runs


def temporal(events, presence, period_ns=PERIOD_NS):
    """Per-episode first-detection latency and age of the newest confirming exposure."""
    rows = []
    for target, present in presence.items():
        for a, b in episodes(present):
            inside_events = [e for e in events if a <= e["tick"] < b]
            latest, first, ages = None, None, []
            for event in inside_events:
                if target in event["hits"]:
                    latest = event["sensor_ns"]
                    if first is None:
                        first = (event["result_ns"] - a * period_ns) / MS
                if latest is not None:
                    ages.append((event["result_ns"] - latest) / MS)
            rows.append(
                dict(
                    target=target,
                    start_tick=a,
                    end_tick=b,
                    duration_ms=(b - a) * period_ns / MS,
                    processed=len(inside_events),
                    first_detection_ms=first,
                    confirmed_ages_ms=ages,
                )
            )
    return rows


def summarize(policy_runs, labels):
    """Pool policy events and episodes across scenes."""
    events = [e for run in policy_runs for e in run["events"]]
    eps = [e for run in policy_runs for e in run["episodes"]]
    counts = {label: dict(tp=0, fn=0) for label in LABELS}
    for run in policy_runs:
        for event in run["events"]:
            for target in event["hits"]:
                counts[labels[run["scene"]][target]]["tp"] += 1
            for target in event["missed"]:
                counts[labels[run["scene"]][target]]["fn"] += 1
    detected = [e["first_detection_ms"] for e in eps if e["first_detection_ms"] is not None]
    ages = [a for e in eps for a in e["confirmed_ages_ms"]]
    seconds = sum(run["ticks"] for run in policy_runs) * PERIOD_NS / 1e9
    return dict(
        frames=len(events),
        frames_per_sequence_second=len(events) / seconds,
        mode_counts={m: sum(e["mode"] == m for e in events) for m in MODES},
        frame_counts={
            label: dict(c, recall=c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else None)
            for label, c in counts.items()
        },
        false_positives=sum(e["fp"] for e in events),
        ignored_partial_detections=sum(e["ignored"] for e in events),
        episodes=len(eps),
        processed_episodes=sum(e["processed"] > 0 for e in eps),
        detected_episodes=len(detected),
        missed_processed_episodes=[
            dict(
                scene=run["scene"],
                target=e["target"],
                ticks=[e["start_tick"], e["end_tick"]],
                processed=e["processed"],
            )
            for run in policy_runs
            for e in run["episodes"]
            if e["processed"] and e["first_detection_ms"] is None
        ],
        first_detection=statistics(detected) if detected else None,
        confirmed_observation_age=statistics(ages) if ages else None,
    )


def timing_model(replay_dir, live_log):
    """Per-mode cost from Pi periodic replays and camera delivery from the live log."""
    costs = {m: [] for m in MODES}
    logs = sorted(Path(replay_dir).glob("periodic-*.frames.jsonl"))
    if len(logs) < 2:
        raise ValueError("Require two periodic Pi replay logs")
    for log in logs:
        for line in log.read_text().splitlines():
            row = json.loads(line)
            costs[row["mode"]].append(row["scheduled_processing_ms"])
    live = [json.loads(s) for s in Path(live_log).read_text().splitlines()]
    delivery = [
        (r["capture_completed_boottime_ns"] - r["sensor_timestamp_ns"]) / MS
        for r in live
        if r["timestamp_valid"]
    ]
    if any(not v for v in costs.values()) or not delivery:
        raise ValueError("Incomplete timing evidence")
    return dict(
        cost_ms={m: float(np.mean(v)) for m, v in costs.items()},
        cost_frames={m: len(v) for m, v in costs.items()},
        delivery_ms=float(np.median(delivery)),
        period_ms=PERIOD_NS / MS,
        replay_logs={p.name: sha256(p) for p in logs},
        live_log_sha256=sha256(live_log),
        caveat="Constant per-mode means; dark-scene delivery latency includes saturated 66.7 ms exposure",
    )


def load_scenes(archive_path, review_root, decision_path, family_path):
    family = read_json(family_path)
    if sha256(archive_path) != family["source_archive_sha256"]:
        raise ValueError("EngDes2 export changed")
    frames, _ = reviewed_frames(review_root, decision_path)
    scenes = []
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        for row, scan in frames:
            if not row["source"].startswith("IMG_"):
                raise ValueError("Require the reserved IMG_ development family")
            matches = [n for n in names if n == f"valid/{row['source']}"]
            if len(matches) != 1:
                raise ValueError("Development source must come from upstream valid")
            with Image.open(io.BytesIO(archive.read(matches[0]))) as im:
                source = np.asarray(im.convert("RGB"))
            if source.shape != (SOURCE, SOURCE, 3):
                raise ValueError("Require 640x640 export")
            if not np.array_equal(render(source, (0, 0, SOURCE)), scan):
                raise ValueError("Rendered full view differs from the reviewed frame")
            targets = [
                dict(
                    label=t["label"],
                    box=t["box"],
                    source_box=[
                        t["box"][0] * SOURCE / OUTPUT[0],
                        t["box"][1] * SOURCE / OUTPUT[1],
                        t["box"][2] * SOURCE / OUTPUT[0],
                        t["box"][3] * SOURCE / OUTPUT[1],
                    ],
                )
                for t in row["provisional_truth"]
            ]
            scenes.append(dict(source=row["source"], image=source, scan=scan, targets=targets))
    return scenes


def evaluate(args):
    base = Path(args.base)
    verify_bundle(base)
    inputs, golden = read_json(base / "inputs.json"), read_json(base / "search-golden.json")
    indoor = [(i, f) for i, f in enumerate(inputs["frames"]) if f["panel"] == "indoor"]
    scenes = load_scenes(args.archive, args.review_root, args.decision, args.family)
    if [f["source"] for _, f in indoor] != [s["source"] for s in scenes]:
        raise ValueError("Bundle indoor frames differ from reviewed scenes")
    native = NativeBalloonRegions(args.library, direct=True)
    timing = timing_model(args.pi_replay, args.live_log)
    cost_ns = {m: round(v * MS) for m, v in timing["cost_ms"].items()}
    delivery_ns = round(timing["delivery_ms"] * MS)
    ticks = SECONDS * 1_000_000_000 // PERIOD_NS
    views = []
    for index, scene in enumerate(scenes):
        wins = [window(k * PERIOD_NS / 1e9, 0.9 * index) for k in range(ticks)]
        truth = [view_truth(scene["targets"], w) for w in wins]
        presence = {
            i: [any(t["target"] == i and t["present"] for t in rows) for rows in truth]
            for i in range(len(scene["targets"]))
        }
        views.append(dict(windows=wins, truth=truth, presence=presence))
    labels = [[t["label"] for t in s["targets"]] for s in scenes]
    results = {}
    for name in args.model:
        predictor = Predictor(base / inputs["models"][name]["path"], allow_unvalidated=True)
        for (frame_index, _), scene in zip(indoor, scenes, strict=True):
            for mode in MODES:
                expected = golden[name]["mser" if mode == "mser_direct" else mode][frame_index]
                observed = process_frame(scene["scan"], predictor, mode, native)
                check_predictions(observed["detections"], expected["detections"])
        static = {m: dict(tp=0, fn=0, fp=0, ignored=0) for m in MODES}
        disagreement = dict(full_only=0, bright_only=0, views=0)
        runs = {"full": [], "periodic": []}
        for index, (scene, view) in enumerate(zip(scenes, views, strict=True)):
            scored = []
            for win, truth in zip(view["windows"], view["truth"], strict=True):
                rgb = render(scene["image"], win)
                tick = {
                    m: score_view(truth, process_frame(rgb, predictor, m, native)["detections"])
                    for m in MODES
                }
                for m, s in tick.items():
                    for key, value in (("tp", len(s["hits"])), ("fn", len(s["missed"]))):
                        static[m][key] += value
                    static[m]["fp"] += s["fp"]
                    static[m]["ignored"] += s["ignored"]
                full, bright = set(tick["mser_direct"]["hits"]), set(tick["mser_bright"]["hits"])
                disagreement["full_only"] += len(full - bright)
                disagreement["bright_only"] += len(bright - full)
                disagreement["views"] += 1
                scored.append(tick)
            for policy in runs:
                events = simulate(policy, scored, cost_ns, delivery_ns)
                runs[policy].append(
                    dict(
                        scene=index,
                        ticks=len(scored),
                        events=events,
                        episodes=temporal(events, view["presence"]),
                    )
                )
        results[name] = dict(
            model_sha256=sha256(base / inputs["models"][name]["path"]),
            static_views=static,
            mode_disagreement=disagreement,
            policies={p: summarize(r, labels) for p, r in runs.items()},
            schedules={
                p: [[[e["tick"], e["mode"][5]] for e in run["events"]] for run in r]
                for p, r in runs.items()
            },
        )
    scene_rows = [
        dict(
            source=s["source"],
            targets=[dict(label=t["label"], box=t["box"]) for t in s["targets"]],
            episodes={str(i): episodes(p) for i, p in v["presence"].items()},
            never_present=[i for i, p in v["presence"].items() if not any(p)],
        )
        for s, v in zip(scenes, views, strict=True)
    ]
    return dict(
        scope=(
            "Synthetic window pans over twelve still development photos; "
            "seen-environment diagnostic, not real motion, occlusion or qualification"
        ),
        trajectory=dict(
            seconds=SECONDS,
            period_ms=PERIOD_NS / MS,
            side_px="368 + 48 sin(2 pi t / 7 + phase)",
            phase="0.9 * scene index",
            present_fraction=PRESENT_FRACTION,
        ),
        timing_model=timing,
        scenes=scene_rows,
        results=results,
        bundle_sha256=sha256(base / "bundle.json"),
        script_sha256=sha256(__file__),
        scheduler_sha256=sha256(Path(__file__).parents[1] / "src/dtr/research_search_schedule.py"),
        decision_sha256=sha256(args.decision),
        native_receipt=read_json(Path(args.library).with_suffix(".json")),
        camera_used=False,
        pi_timing_measured=False,
        temporal_recall_measured="synthetic_pan_only",
        hard_reacquisition_deadline_proven=False,
        training_performed=False,
        test_evaluated=False,
        deployment_approved=False,
        flight_commands=None,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "base",
        "library",
        "archive",
        "review-root",
        "decision",
        "family",
        "pi-replay",
        "live-log",
        "output",
    ):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--model", action="append", required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    cv2.setNumThreads(1)
    report = evaluate(args)
    write_json(args.output, report)
    for name, result in report["results"].items():
        print(name, json.dumps(result["static_views"]), json.dumps(result["mode_disagreement"]))
        for policy, summary in result["policies"].items():
            print(
                policy,
                json.dumps({k: v for k, v in summary.items() if k != "missed_processed_episodes"}),
            )


if __name__ == "__main__":
    main()
