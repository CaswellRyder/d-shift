"""Profile search only after normal timing trials; instrumentation is not FPS."""

import argparse
import cProfile
from pathlib import Path
import platform
import pstats
import sys

import cv2
import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root, out = Path(args.base).resolve(), Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    sys.path.insert(0, str(root))
    from dtr.data import read_json, sha256, write_json
    from pi_red_blue_bench import verify_bundle
    from research_balloon_search import experimental

    verify_bundle(root)
    inputs = read_json(root / "inputs.json")
    images = [np.asarray(Image.open(root / r["path"]).convert("RGB")) for r in inputs["frames"]]
    cv2.setNumThreads(1)
    profile = cProfile.Profile()
    profile.enable()
    counts = [len(experimental(rgb, "mser")) for rgb in images]
    profile.disable()
    stats = pstats.Stats(profile)
    entries = []
    for (file, line, name), (primitive, calls, total, cumulative, _) in stats.stats.items():
        entries.append(
            dict(
                file=file,
                line=line,
                function=name,
                calls=calls,
                primitive_calls=primitive,
                total_seconds=total,
                cumulative_seconds=cumulative,
            )
        )
    report = dict(
        machine=platform.machine(),
        opencv=cv2.__version__,
        frames=len(images),
        proposals=counts,
        profile_total_seconds=stats.total_tt,
        entries=sorted(entries, key=lambda r: -r["cumulative_seconds"]),
        bundle_sha256=sha256(root / "bundle.json"),
        script_sha256=sha256(__file__),
        scope="Instrumented search-only profile, not normal FPS or accuracy",
        test_evaluated=False,
        deployment_approved=False,
        camera_used=False,
    )
    write_json(out, report)
    print(report)


if __name__ == "__main__":
    main()
