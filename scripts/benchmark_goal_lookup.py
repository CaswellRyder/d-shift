"""Paired all-frame proposal parity/timing; inputs predecoded, no camera or model."""
import argparse
from pathlib import Path
import platform
import time

import cv2
import numpy as np
from PIL import Image

from dtr.data import read_json,sha256,write_json
from dtr.vision import proposals


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    base,out=Path(args.base).resolve(),Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    manifest=read_json(base/"frames.json")
    if manifest["split"]!="development_validation":
        raise ValueError("Development inputs only")
    cv2.setNumThreads(1)
    times={"balloon_components":[],"goal_lut":[]}
    for index,row in enumerate(manifest["frames"]):
        path=(base/row["path"]).resolve()
        if not path.is_relative_to(base) or sha256(path)!=row["sha256"]:
            raise ValueError("Input checksum/path mismatch")
        with Image.open(path) as im:
            rgb=np.array(im.convert("RGB"))
        if index==0:
            for profile in times:
                for _ in range(3):
                    proposals(rgb,"goal",profile=profile)
        results={}
        order=list(times) if index%2==0 else list(times)[::-1]
        for profile in order:
            start=time.perf_counter()
            results[profile]=proposals(rgb,"goal",profile=profile)
            times[profile].append((time.perf_counter()-start)*1000)
        if results["balloon_components"]!=results["goal_lut"]:
            raise ValueError(f"Proposal parity failed on {row['path']}")
        if index%100==0:
            print("Frames",index+1,"/",len(manifest["frames"]),flush=True)
    report=dict(frames=len(manifest["frames"]),exact_proposal_parity=True,host=platform.node(),machine=platform.machine(),
                frame_manifest_sha256=sha256(base/"frames.json"),
                results={k:dict(mean_ms=float(np.mean(v)),p95_ms=float(np.percentile(v,95))) for k,v in times.items()},
                scope="Goal proposal search only; excludes frame decoding, camera, model and one-time LUT load",
                deployment_approved=False,flight_commands=None)
    write_json(out,report)
    print(report,flush=True)


if __name__=="__main__":
    main()
