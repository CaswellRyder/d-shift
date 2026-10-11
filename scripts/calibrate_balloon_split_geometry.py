"""Calibrate an oval-child shape bound from admitted TRAIN annotations only."""
import argparse
from pathlib import Path

import numpy as np

from dtr.data import sha256, write_json
from scripts.audit_balloon_search_stages import original_frames, indoor_frames, training_source


def calibrate(panels):
    rows, seen = [], set()
    for panel, frames in panels.items():
        for source, _, targets in frames:
            training_source(source)
            for target in targets:
                key = (panel, source, target["annotation_id"])
                if key in seen:
                    raise ValueError("Duplicate training annotation")
                seen.add(key)
                x, y, z, w = target["box"]
                if (target["label"] not in ("red_balloon", "blue_balloon")
                        or not np.isfinite([x,y,z,w]).all()
                        or not 0 <= x < z <= 320 or not 0 <= y < w <= 240):
                    raise ValueError("Invalid training annotation")
                ratio = max((z-x)/(w-y), (w-y)/(z-x))
                rows.append(dict(panel=panel, source=source, annotation_id=target["annotation_id"],
                                 label=target["label"], aspect_ratio=ratio))
    if not rows:
        raise ValueError("No admitted training annotations")
    # Fixed protocol: no development threshold sweep. The upper cap is the
    # existing candidate() aspect limit; the floor preserves the original bound.
    quantile = float(np.quantile([r["aspect_ratio"] for r in rows], .95))
    return dict(schema="balloon-split-geometry-v1", quantile=.95,
                raw_aspect_quantile=quantile, max_aspect=max(1.5, min(3., quantile)),
                control_max_aspect=1.5, candidate_max_aspect=3., annotations=rows,
                annotation_count=len(rows), source_count=len({(r["panel"],r["source"]) for r in rows}),
                training_only=True, test_evaluated=False, deployment_approved=False,
                independent_recording_sessions_verified=False,
                scope="Training-annotation shape calibration, not classifier training or qualification")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "original-archive", "queue", "review", "indoor-archive", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = calibrate(dict(
        original=original_frames(args.manifest, args.original_archive),
        indoor=indoor_frames(args.queue, args.review, args.manifest, args.indoor_archive)))
    report.update(script_sha256=sha256(__file__),
                  loader_sha256=sha256("scripts/audit_balloon_search_stages.py"),
                  inputs={name: sha256(path) for name,path in dict(
                      manifest=args.manifest, original_archive=args.original_archive,
                      queue=args.queue/"review.json", review=args.review,
                      indoor_archive=args.indoor_archive).items()})
    write_json(args.output, report)
    print({k:v for k,v in report.items() if k not in ("annotations", "inputs")})
