"""Static-target persistence simulation from frozen predictions and Pi timings.

Each development photo is held unchanged for three seconds. This tests search
dispatch and systematic misses, NOT motion tracking, flight, or live throughput.
"""

import argparse
from collections import defaultdict
import math
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from dtr.research_search_schedule import SearchSchedule
from dtr.tracking import iou
from scripts.compare_pi_search_paths import validate_identity
from scripts.pi_balloon_search_bench import check_predictions

MS = 1_000_000
POLICIES = ("full", "pixel", "recovery")


def matched_targets(truth, detections):
    matched, false_positives = set(), 0
    for detection in sorted(detections, key=lambda d: -d["score"]):
        if not detection["accepted"] or detection["label"] not in ("red_balloon", "blue_balloon"):
            continue
        overlap, index = max(((iou(t["box"], detection["box"]), i)
                              for i, t in enumerate(truth)
                              if i not in matched and t["label"] == detection["label"]),
                             default=(0, -1))
        if overlap >= .5:
            matched.add(index)
        else:
            false_positives += 1
    return matched, false_positives


def dispatch(policy, costs, duration_ns=3_000_000_000, period_ns=100_000_000,
             delivery_ns=125_000_000):
    """Simulate serial fresh-frame capture, with explicit non-preemptive cost.

    Pixel steps reuse the existing full-search scheduler's between-full slots.
    No detections are cached or carried forward by this simulation.
    """
    if policy not in POLICIES:
        raise ValueError("Unknown policy")
    if any(type(n) is not int or n <= 0 for n in (duration_ns, period_ns)):
        raise ValueError("Positive integer duration and period required")
    if type(delivery_ns) is not int or delivery_ns < 0:
        raise ValueError("Nonnegative integer delivery delay required")
    if set(costs) != {"mser", "pixel2"} or any(type(v) is not int or v <= 0 for v in costs.values()):
        raise ValueError("Positive integer processing costs required")
    schedule, result, events = SearchSchedule(), 0, []
    while True:
        tick = max(0, (result - delivery_ns + period_ns - 1) // period_ns)
        sensor = tick * period_ns
        start = sensor + delivery_ns
        if start >= duration_ns:
            break
        if policy == "pixel":
            method, reason = "pixel2", "pixel_only"
        else:
            decision = schedule.begin(start, force_full=policy == "full")
            method = "pixel2" if decision["mode"] == "mser_bright" else "mser"
            reason = "pixel_between_full" if method == "pixel2" else decision["reason"]
        result = start + costs[method]
        if policy != "pixel":
            schedule.finish(result)
        if result > duration_ns:
            break  # Don't credit results that arrive after the evaluation window.
        events.append(dict(sensor_ns=sensor, start_ns=start, result_ns=result,
                           method=method, reason=reason))
    return events


def target_metrics(events, matches, target, duration_ns):
    hits = [e for e in events if target in matches[e["method"]]]
    if not hits:
        return dict(confirmations=0, first_detection_ms=None, max_confirmation_gap_ms=None,
                    end_confirmation_age_ms=None)
    gaps = [(b["result_ns"] - a["result_ns"]) / MS for a, b in zip(hits, hits[1:])]
    return dict(confirmations=len(hits), first_detection_ms=hits[0]["result_ns"] / MS,
                max_confirmation_gap_ms=max(gaps) if gaps else None,
                end_confirmation_age_ms=(duration_ns - hits[-1]["sensor_ns"]) / MS)


def evaluate(base, golden_path, replay):
    base, replay = Path(base), Path(replay)
    golden, report = read_json(golden_path), read_json(replay / "report.json")
    validate_identity(golden["identity"], report["identity"])
    inputs = read_json(base / "inputs.json")
    if (sha256(base / "inputs.json") != golden["identity"]["inputs_sha256"]
            or sha256(golden_path) != report["golden_sha256"] or report["machine"] != "armv6l"):
        raise ValueError("Input, golden or Pi identity mismatch")
    costs = defaultdict(lambda: defaultdict(list))
    for trial in report["trials"]:
        if trial["method"] not in ("mser", "pixel2"):
            continue
        path = replay / f"{trial['trial']}-{trial['method']}.json"
        if sha256(path) != trial["rows_sha256"]:
            raise ValueError("Changed raw timings")
        rows = read_json(path)["rows"]
        if [r["index"] for r in rows] != list(range(len(inputs["frames"]))):
            raise ValueError("Incomplete timing coverage")
        for row, prediction in zip(rows, golden["predictions"][trial["method"]], strict=True):
            check_predictions(row["detections"], prediction)
            value = row["processing_ms"]
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Invalid processing cost")
            costs[row["index"]][trial["method"]].append(value)
    duration = 3_000_000_000
    scenes = []
    for index, frame in enumerate(inputs["frames"]):
        if any(len(costs[index][m]) != 2 for m in ("mser", "pixel2")):
            raise ValueError("Require two timing repeats per method and scene")
        timing = {m: int(round(np.mean(costs[index][m]) * MS)) for m in ("mser", "pixel2")}
        scored = {m: matched_targets(frame["truth"], golden["predictions"][m][index])
                  for m in timing}
        policies = {}
        for policy in POLICIES:
            events = dispatch(policy, timing, duration_ns=duration)
            policies[policy] = dict(
                processed=len(events), projected_fps=len(events) / 3,
                full_searches=sum(e["method"] == "mser" for e in events),
                false_positive_observations=sum(scored[e["method"]][1] for e in events),
                targets=[dict(label=t["label"], target=i, **target_metrics(
                    events, {m: s[0] for m, s in scored.items()}, i, duration))
                    for i, t in enumerate(frame["truth"])])
        scenes.append(dict(index=index, panel=frame["panel"], policies=policies))
    summary = {}
    for policy in POLICIES:
        targets = [t for s in scenes for t in s["policies"][policy]["targets"]]
        per_color = {}
        for color in ("red_balloon", "blue_balloon"):
            rows = [t for t in targets if t["label"] == color]
            detected = [t for t in rows if t["confirmations"]]
            per_color[color] = dict(targets=len(rows), acquired=len(detected), never_acquired=len(rows)-len(detected))
        latencies = [t["first_detection_ms"] for t in targets if t["confirmations"]]
        gaps = [t["max_confirmation_gap_ms"] for t in targets if t["max_confirmation_gap_ms"] is not None]
        summary[policy] = dict(
            colors=per_color,
            projected_fps=sum(s["policies"][policy]["processed"] for s in scenes) / (3 * len(scenes)),
            first_detection_p95_ms=float(np.percentile(latencies, 95)) if latencies else None,
            maximum_confirmation_gap_ms=max(gaps) if gaps else None,
            false_positive_observations=sum(s["policies"][policy]["false_positive_observations"] for s in scenes))
    return dict(summary=summary, scenes=scenes, model=golden["identity"]["model"],
                golden_sha256=sha256(golden_path), replay_sha256=sha256(replay / "report.json"),
                script_sha256=sha256(__file__), scheduler_sha256=sha256(
                    Path(__file__).resolve().parents[1] / "src/dtr/research_search_schedule.py"),
                scope="Three-second repeated stills; not real tracking, motion or new hardware timing",
                simulation=True, delivery_delay_assumption_ms=125, sensor_period_assumption_ms=100,
                camera_used=False, test_evaluated=False, deployment_approved=False, flight_commands=None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "golden", "replay", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    result = evaluate(args.base, args.golden, args.replay)
    write_json(args.output, result)
    print(result["summary"])


if __name__ == "__main__":
    main()
