"""Paired tracking ablation with fixed weights, equal inputs, alternating order.

--fixed-clock audits semantic equivalence with inference-time expiry removed; it
is not a real-time result. Normal mode retains actual inference-time label expiry.
"""
import argparse
import json
from pathlib import Path
import platform
import sys
import time

import cv2
import numpy as np
from PIL import Image

from dtr.data import read_json,sha256,write_json
from dtr.runtime import Predictor
from dtr.temporal import TemporalVision
from pi_temporal_bench import translated,select_clips


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base",required=True)
    parser.add_argument("--model",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--fixed-clock",action="store_true")
    parser.add_argument("--lookup",action="store_true",help="Add a third native-difference + lookup variant")
    args=parser.parse_args()
    out,base=Path(args.output),Path(args.base).resolve()
    if out.exists() or out.with_suffix(".frames.jsonl").exists():
        raise FileExistsError(out)
    bundle=Path(__file__).resolve().parent
    if (bundle/"bundle.json").exists():
        receipt=read_json(bundle/"bundle.json")
        for name,digest in receipt["files"].items():
            path=(bundle/name).resolve()
            if not path.is_relative_to(bundle) or sha256(path)!=digest:
                raise ValueError("Bundle changed")
        for name,digest in receipt.get("replay_base_sha256",{}).items():
            if sha256(base/name)!=digest:
                raise ValueError("Base helper changed")
    manifest=read_json(base/"frames.json")
    if manifest["split"]!="development_validation":
        raise ValueError("Development frames only")
    sys.path.insert(0,str(base))
    from pi_compare import localization_counts,metrics
    model=Predictor(args.model,True)
    classes=[c for c in model.metadata["classes"] if c!="background"]
    cv2.setNumThreads(1)
    rows=select_clips(manifest["frames"],classes)
    variants={"numpy":dict(difference_backend="numpy"), "native":dict(difference_backend="native")}
    if args.lookup:
        variants["native_lookup"]=dict(difference_backend="native",proposal_profile="goal_lut")
    trackers={k:TemporalVision(model,budget=4,**v) for k,v in variants.items()}
    states={k:dict(times=[],counts={c:dict(tp=0,fp=0,fn=0) for c in classes},calls=0,scans=0,
                   max_age_ms=0.,over_100ms=0) for k in variants}
    mismatches={k:0 for k in variants if k!="numpy"}
    real_clock=time.perf_counter
    with out.with_suffix(".frames.jsonl").open("x") as stream:
        for clip,row in enumerate(rows):
            path=(base/row["path"]).resolve()
            if not path.is_relative_to(base) or sha256(path)!=row["sha256"]:
                raise ValueError("Frame checksum/path mismatch")
            with Image.open(path) as im:
                rgb=np.array(im.convert("RGB"))
            # Warm search, model, and optional table outside timed work. Reset labels.
            for tracker in trackers.values():
                tracker.reset()
                tracker.observe(rgb,0.)
                tracker.reset()
            for index in range(20):
                frame,truth=translated(rgb,row["truth"],index)
                order=list(trackers) if (clip+index)%2==0 else list(trackers)[::-1]
                results={}
                for name in order:
                    start=real_clock()
                    try:
                        if args.fixed_clock:
                            time.perf_counter=lambda:0.
                        result=trackers[name].observe(frame,index*.1)
                    finally:
                        time.perf_counter=real_clock
                    elapsed=(real_clock()-start)*1000
                    results[name]=result
                    state=states[name]
                    state["times"].append(elapsed)
                    state["over_100ms"]+=elapsed>100
                    state["calls"]+=result["inference_calls"]
                    state["scans"]+=result["full_scan"]
                    accepted=[r for r in result["observations"] if r["accepted"]]
                    for item in accepted:
                        state["max_age_ms"]=max(state["max_age_ms"],item["classification_age_ms"])
                    for label in classes:
                        counts=localization_counts([r["box"] for r in accepted if r["label"]==label],
                                                   [r["box"] for r in truth if r["label"]==label])
                        for key in counts:
                            state["counts"][label][key]+=counts[key]
                    stream.write(json.dumps(dict(variant=name,clip=clip,index=index,outer_ms=elapsed,**result))+"\n")
                if args.fixed_clock:
                    for name in mismatches:
                        mismatches[name]+=results[name]!=results["numpy"]
            print("Clip",clip+1,"/",len(rows),flush=True)
    reports={}
    for name,state in states.items():
        counts={k:sum(v[k] for v in state["counts"].values()) for k in ("tp","fp","fn")}
        reports[name]=dict(mean_ms=float(np.mean(state["times"])),p95_ms=float(np.percentile(state["times"],95)),
                           processing_fps=None if args.fixed_clock else 1000/float(np.mean(state["times"])),
                           micro=metrics(counts),classes={k:metrics(v) for k,v in state["counts"].items()},
                           inference_calls=state["calls"],full_scans=state["scans"],over_100ms=state["over_100ms"],
                           max_emitted_age_ms=state["max_age_ms"])
    report=dict(results=reports,model_sha256=sha256(args.model),frames_per_variant=len(rows)*20,
                machine=platform.machine(),host=platform.node(),fixed_clock=args.fixed_clock,
                exact_output_mismatches=mismatches if args.fixed_clock else None,
                scope="Synthetic translation of 12 development stills; not camera FPS or field accuracy",
                timing_scope="Full processing call; excludes generation, loading, startup, camera and writes",
                deployment_approved=False,flight_commands=None)
    write_json(out,report)
    print(json.dumps(report,indent=2),flush=True)
    if args.fixed_clock and any(mismatches.values()):
        raise SystemExit("Semantic parity failed")


if __name__=="__main__":
    main()
