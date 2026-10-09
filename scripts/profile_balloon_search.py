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
    parser.add_argument("--search", choices=("mser", "mser_fast", "mser_direct"), default="mser")
    parser.add_argument("--region-library")
    args = parser.parse_args()
    root, out = Path(args.base).resolve(), Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    sys.path.insert(0, str(root))
    from dtr.data import read_json, sha256, write_json
    from pi_red_blue_bench import verify_bundle
    from research_balloon_search import experimental

    verify_bundle(root)
    native = None
    if args.search in ("mser_fast", "mser_direct"):
        if not args.region_library:
            parser.error("Fast search requires --region-library")
        from dtr.native_balloon_regions import NativeBalloonRegions
        from research_balloon_search_fast import search_fast

        native = NativeBalloonRegions(args.region_library, direct=args.search == "mser_direct")
        if read_json(native.path.with_suffix(".json"))["source_sha256"] != sha256(
            root / "dtr/native_balloon_regions.c"
        ):
            raise ValueError("Native region source receipt mismatch")
    elif args.region_library:
        parser.error("Region library requires fast search")
    inputs = read_json(root / "inputs.json")
    images = [np.asarray(Image.open(root / r["path"]).convert("RGB")) for r in inputs["frames"]]
    cv2.setNumThreads(1)
    profile = cProfile.Profile()
    profile.enable()
    counts = [
        len(search_fast(rgb, native) if native else experimental(rgb, "mser")) for rgb in images
    ]
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
        search=args.search,
        region_library_sha256=sha256(native.path) if native else None,
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
