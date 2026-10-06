"""Observational Pi benchmark: controlled replay or real camera, no actuation.

Replay translates public development stills. It measures engineering behavior,
NOT video/flight generalization. Camera mode has no labels and reports no accuracy.
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

from dtr.data import read_json, sha256, write_json
from dtr.runtime import Predictor
from dtr.temporal import TemporalVision
from dtr.vision import observe


def summary(records):
    times = [r["processing_ms"] for r in records]
    ages = [o["classification_age_ms"] for r in records for o in r["observations"]
            if o["accepted"] and "classification_age_ms" in o]
    return dict(frames=len(records), mean_ms=float(np.mean(times)),
                p95_ms=float(np.percentile(times,95)), processing_fps=1000/float(np.mean(times)),
                inference_calls=sum(r["inference_calls"] for r in records),
                full_scans=sum(r["full_scan"] for r in records),
                over_100ms=sum(t>100 for t in times), max_emitted_label_age_ms=max(ages,default=None))


def select_clips(rows, classes):
    """First two distinct frames containing each class; deterministic, not scored selection."""
    picked, used = [], set()
    for label in classes:
        candidates = [r for r in rows if r["path"] not in used
                      and any(t["label"] == label for t in r["truth"])]
        for row in candidates[:2]:
            picked.append(row)
            used.add(row["path"])
    return picked


def translated(rgb, truth, index):
    # 8 slow-motion frames, 2 missing frames, 10 return frames; 1 pixel / 100 ms.
    if index in (8,9):
        return np.zeros_like(rgb), []
    offset = index if index < 8 else index-10
    matrix = np.float32([[1,0,offset],[0,1,0]])
    frame = cv2.warpAffine(rgb,matrix,(rgb.shape[1],rgb.shape[0]))
    shifted = []
    for item in truth:
        a,b,c,d = item["box"]
        a,c = max(0,a+offset),min(rgb.shape[1],c+offset)
        if c > a:
            shifted.append(dict(label=item["label"],box=[a,b,c,d]))
    return frame,shifted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base",required=True)
    parser.add_argument("--model",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--mode",choices=["replay","camera"],default="replay")
    parser.add_argument("--budget",type=int,default=4)
    parser.add_argument("--profile",choices=("balloon_components","goal_lut"),default="balloon_components")
    parser.add_argument("--difference-backend",choices=("numpy","native"),default="native")
    parser.add_argument("--policy",choices=["both","temporal"],default="both")
    parser.add_argument("--background-refresh",type=float,default=1.8)
    parser.add_argument("--frames",type=int,default=60,help="Camera frames, bounded 1..600")
    parser.add_argument("--camera-rotation",type=int,choices=[0,180],default=180)
    args = parser.parse_args()
    base, output = Path(args.base),Path(args.output)
    if output.exists() or output.with_suffix(".frames.jsonl").exists():
        raise FileExistsError(output)
    if not 1 <= args.frames <= 600:
        parser.error("frames must be 1..600")
    # New transferable bundles bind source, weights, scoring helper and input list.
    # Older snapshots remain runnable; this does not modify them.
    bundle = Path(__file__).resolve().parent
    receipt_path = bundle / "bundle.json"
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        for relative, digest in receipt["files"].items():
            path = (bundle / relative).resolve()
            if not path.is_relative_to(bundle) or sha256(path) != digest:
                raise ValueError("Replay bundle changed or escaped root")
        for name, digest in receipt.get("replay_base_sha256", {}).items():
            path = (base / name).resolve()
            if not path.is_relative_to(base.resolve()) or sha256(path) != digest:
                raise ValueError("Frozen replay base changed or escaped root")
    if args.mode == "replay":
        manifest = read_json(base / "frames.json")
        if manifest.get("split") != "development_validation":
            raise ValueError("Replay requires the frozen development set")
        for row in manifest["frames"]:
            path = (base / row["path"]).resolve()
            if not path.is_relative_to(base.resolve()) or sha256(path) != row["sha256"]:
                raise ValueError("Replay image changed or escaped root")
    cv2.setNumThreads(1)
    sys.path.append(str(base.resolve()))
    from pi_compare import load_baseline, localization_counts, metrics
    predictor = Predictor(args.model,allow_unvalidated=True)
    classes = [c for c in predictor.metadata["classes"] if c != "background"]
    tracker = TemporalVision(predictor,budget=args.budget,background_refresh_interval=args.background_refresh,
                             proposal_profile=args.profile,difference_backend=args.difference_backend)
    report = dict(host=platform.node(),machine=platform.machine(),mode=args.mode,
                  model_sha256=sha256(args.model),base_manifest_sha256=sha256(base/"frames.json"),
                  budget=args.budget,scan_interval=.5,refresh_interval=.6,label_ttl=1.,
                  proposal_profile=args.profile,
                  difference_backend=args.difference_backend,
                  background_refresh_interval=args.background_refresh,
                  flight_commands=None,deployment_approved=False,test_evaluated=False)
    records = {"baseline":[],"stateless":[],"temporal":[]}
    if args.policy == "temporal" or args.mode == "camera":
        records.pop("stateless")
        records.pop("baseline")
    with output.with_suffix(".frames.jsonl").open("x") as stream:
        if args.mode == "replay":
            baseline = load_baseline(base/"baseline.py")
            detector = baseline.Detector(baseline.TARGET_COLOR_RGB,baseline.COLOR_THRESHOLD,
                                         baseline.MIN_SHAPE_SIZE,backend="numpy")
            rows = select_clips(read_json(base/"frames.json")["frames"],classes)
            counts = {method:{label:dict(tp=0,fp=0,fn=0)
                              for label in (['yellow_localization'] if method == "baseline" else classes+['yellow_localization'])}
                      for method in records}
            for clip,row in enumerate(rows):
                rgb = np.array(Image.open(base/row["path"]).convert("RGB"))
                tracker.reset()
                # Both paths warmed before the timed clip; tracker reset prevents leaked labels.
                for _ in range(3):
                    predictor.predict(rgb)
                    detector.detect(rgb)
                for index in range(20):
                    frame,truth = translated(rgb,row["truth"],index)
                    methods = ("baseline","stateless","temporal")
                    if (clip+index)%2:
                        methods = methods[::-1]
                    for method in methods:
                        if method not in records:
                            continue
                        operation_start = time.perf_counter()
                        if method == "baseline":
                            start = time.perf_counter()
                            boxes,_ = detector.detect(frame)
                            observations = [dict(label="yellow_goal",accepted=True,
                                                 box=[int(a),int(b),int(c)+1,int(d)+1]) for a,b,c,d in boxes]
                            result = dict(observations=observations,processing_ms=(time.perf_counter()-start)*1000,
                                          inference_calls=0,full_scan=True,flight_commands=None)
                        elif method == "temporal":
                            result = tracker.observe(frame,now=index*.1)
                        else:
                            result = observe(frame,predictor,limit=12,profile=args.profile,duplicate_policy="nested")
                            result.update(inference_calls=len(result["observations"]),full_scan=True)
                        result["core_processing_ms"] = result["processing_ms"]
                        result["processing_ms"] = (time.perf_counter()-operation_start)*1000
                        result.update(clip=clip,source=row["path"],frame_index=index,method=method,
                                      truth=truth,phase="missing" if index in (8,9) else "present")
                        accepted = [o for o in result["observations"] if o["accepted"]]
                        for label in counts[method]:
                            def matches(name):
                                return name.startswith("yellow_") if label == "yellow_localization" else name == label
                            got = localization_counts([o["box"] for o in accepted if matches(o["label"])],
                                                       [t["box"] for t in truth if matches(t["label"])])
                            for key in got:
                                counts[method][label][key] += got[key]
                        records[method].append(result)
                        stream.write(json.dumps(result)+"\n")
                print("Clip",clip+1,"/",len(rows),flush=True)
            recovery = {}
            for method,items in records.items():
                if method == "baseline":
                    recovery[method] = None  # Cannot identify classes.
                    continue
                delays = []
                for clip in range(len(rows)):
                    returned = [r for r in items if r["clip"] == clip and r["frame_index"] >= 10]
                    correct = [r for r in returned if any(
                        localization_counts([o["box"] for o in r["observations"] if o["accepted"] and o["label"] == label],
                                            [t["box"] for t in r["truth"] if t["label"] == label])["tp"] > 0
                        for label in classes)]
                    delays.append((correct[0]["frame_index"]-10)*100 if correct else None)
                recovery[method] = delays
            report.update(scope="Synthetic translation of development stills; NOT real video accuracy",
                          nominal_frame_interval_ms=100, clips=[r["path"] for r in rows],
                          timing_scope="processing only; nominal clock does not imply 10 Hz real-time",
                          results={method:{**summary(items),"metrics":{k:metrics(v) for k,v in counts[method].items()},
                                   "first_correct_class_after_return_nominal_ms_by_clip":recovery[method],
                                   "missing_frame_false_positives":sum(o["accepted"] for r in items if r["phase"] == "missing" for o in r["observations"])}
                                   for method,items in records.items()})
        else:
            from libcamera import Transform
            from picamera2 import Picamera2
            camera = Picamera2()
            camera.configure(camera.create_video_configuration(main={"size":(320,240),"format":"RGB888"},
                                                              controls={"FrameRate":10},buffer_count=2,
                                                              transform=Transform(hflip=args.camera_rotation==180,
                                                                                  vflip=args.camera_rotation==180)))
            report.update(camera_rotation_degrees=args.camera_rotation,
                          configured_transform=str(camera.camera_configuration()["transform"]),
                          coordinate_frame="orientation-corrected camera pixels")
            stamps = []
            camera.start()
            try:
                time.sleep(1)
                start = time.monotonic()
                for index in range(args.frames):
                    request = camera.capture_request()
                    try:
                        rgb = cv2.cvtColor(request.make_array("main"),cv2.COLOR_BGR2RGB)
                        sensor = request.get_metadata().get("SensorTimestamp")
                    finally:
                        request.release()
                    operation_start = time.perf_counter()
                    result = tracker.observe(rgb)
                    result["core_processing_ms"] = result["processing_ms"]
                    result["processing_ms"] = (time.perf_counter()-operation_start)*1000
                    result.update(frame_index=index,sensor_timestamp=sensor)
                    stamps.append(sensor)
                    records["temporal"].append(result)
                    stream.write(json.dumps(result)+"\n")
                    if index in (0,args.frames-1):
                        canvas = rgb.copy()
                        for row in result["observations"]:
                            if row["accepted"]:
                                a,b,c,d = row["box"]
                                cv2.rectangle(canvas,(a,b),(c,d),(0,255,0),1)
                                cv2.putText(canvas,row["label"],(a,max(12,b-3)),cv2.FONT_HERSHEY_SIMPLEX,.35,(0,255,0),1)
                        Image.fromarray(canvas).save(output.with_name(output.stem+f"-{index:04d}.png"))
                wall = time.monotonic()-start
            finally:
                camera.stop()
                camera.close()
            report.update(scope="Unlabeled live camera timing only; no accuracy claim",
                          result=summary(records["temporal"]),wall_seconds=wall,observed_loop_fps=args.frames/wall,
                          increasing_sensor_timestamps=all(a is not None and b is not None and b>a for a,b in zip(stamps,stamps[1:])))
    write_json(output,report)
    print(json.dumps(report,indent=2),flush=True)


if __name__ == "__main__":
    main()
