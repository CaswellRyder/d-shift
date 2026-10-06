"""Audit shape-only evidence on saved development predictions; no model/flight writes."""
import argparse
from collections import Counter
import json
from pathlib import Path

from dtr.data import read_json, sha256, write_json
from dtr.goal_evidence import SHAPES, goal_candidate, goal_evidence
from dtr.vision import suppress_duplicates
from pi_compare import localization_counts, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations",required=True)
    parser.add_argument("--metadata",required=True)
    parser.add_argument("--truth",required=True)
    parser.add_argument("--output",required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() or output.with_suffix(".frames.jsonl").exists():
        raise FileExistsError(output)
    classes = read_json(args.metadata)["classes"]
    truth = {r["path"]:r["truth"] for r in read_json(args.truth)["frames"]}
    names = ("strict_shape","shape_evidence","strict_orange","orange_investigation_pool")
    totals = {name:dict(tp=0,fp=0,fn=0) for name in names}
    counts,seen = Counter(),set()
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.with_suffix(".frames.jsonl").open("x") as stream:
        for line in Path(args.observations).read_text().splitlines():
            frame = json.loads(line)
            key = frame["frame"]
            if key in seen or key not in truth:
                raise ValueError("Duplicate or unmatched frame")
            seen.add(key)
            pools = {name:[] for name in names}
            annotated = []
            for original in frame["observations"]:
                evidence = goal_evidence(original["scores"],classes)
                row = dict(original,goal_evidence=evidence)
                # Saved stills have no measured acquisition age. Zero here is an
                # explicitly hypothetical fresh-frame audit, not a live-age claim.
                selection = goal_candidate(row,"orange",measurement_age_ms=0,allow_unknown_color=True)
                annotated.append(dict(row,orange_investigation=selection))
                counts[evidence["status"]] += 1
                if original.get("suppressed",False):
                    continue
                if original["accepted"]:
                    shape = original["label"].split("_")[1]
                    entry = dict(box=original["box"],label=shape,score=evidence["shape_scores"][shape],accepted=True)
                    pools["strict_shape"].append(entry)
                    if original["label"].startswith("orange_"):
                        pools["strict_orange"].append(entry)
                        pools["orange_investigation_pool"].append(entry)
                if evidence["shape"] != "unknown":
                    entry = dict(box=original["box"],label=evidence["shape"],score=evidence["shape_score"],accepted=True)
                    pools["shape_evidence"].append(entry)
                    if selection["investigation_candidate"]:
                        pools["orange_investigation_pool"].append(entry)
                        counts["additional_orange_investigation_candidates"] += 1
            for name,pool in pools.items():
                kept = [r for r in suppress_duplicates(pool,"nested") if r["accepted"]]
                for shape in SHAPES:
                    expected = [r["box"] for r in truth[key] if r["label"].endswith("_"+shape)
                                and ("orange" not in name or r["label"].startswith("orange_"))]
                    got = localization_counts([r["box"] for r in kept if r["label"] == shape],expected)
                    for metric in got:
                        totals[name][metric] += got[metric]
            stream.write(json.dumps(dict(frame=key,observations=annotated))+"\n")
    if seen != set(truth):
        raise ValueError("Incomplete frame coverage")
    report = dict(frames=len(seen),counts=dict(counts),metrics={k:metrics(v) for k,v in totals.items()},
                  scope="Reused development stills; shape-only score ignores color; investigation pool is NOT confirmed detection",
                  matching="Same shape, IoU>=0.5; shape-based nested duplicate suppression for every pool",
                  measurement_age="Hypothetical zero for offline policy audit; no physical range or live-age measurement",
                  threshold=.8,flight_commands=None,deployment_approved=False,test_evaluated=False,
                  inputs={k:dict(path=getattr(args,k),sha256=sha256(getattr(args,k)))
                          for k in ("observations","metadata","truth")})
    write_json(output,report)
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
