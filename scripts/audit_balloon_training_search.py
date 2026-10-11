"""Compare fixed search policies on admitted training positives, not full accuracy."""
import argparse
from pathlib import Path
import time

import numpy as np

from dtr.data import sha256, write_json
from dtr.tracking import iou
from scripts.audit_balloon_search_stages import original_frames, indoor_frames
from scripts.research_balloon_search import experimental


def compare(frames):
    results = {}
    for variant in ("mser", "mser_instance_split", "mser_opening_split"):
        rows, timings, calls = [], [], []
        for source, rgb, truth in frames:
            start = time.perf_counter()
            candidates = experimental(rgb, variant)
            timings.append((time.perf_counter()-start)*1000)
            calls.append(len(candidates))
            for t in truth:
                group = 0 if t["label"] == "red_balloon" else 1
                overlap = max((iou(t["box"], c["box"]) for c in candidates
                               if c["color_group"] == group), default=0.)
                rows.append(dict(source=source, annotation_id=t["annotation_id"],
                                 label=t["label"], best_iou=overlap, covered=overlap >= .5))
        results[variant] = dict(
            counts={label: dict(covered=sum(r["covered"] for r in rows if r["label"] == label),
                                total=sum(r["label"] == label for r in rows))
                    for label in ("red_balloon", "blue_balloon")},
            desktop_mean_search_ms=float(np.mean(timings)), mean_candidates=float(np.mean(calls)),
            rows=rows)
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "original-archive", "queue", "review", "indoor-archive", "output"):
        p.add_argument(f"--{name}", required=True, type=Path)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    panels = dict(original=compare(list(original_frames(a.manifest, a.original_archive))),
                  indoor=compare(list(indoor_frames(a.queue, a.review, a.manifest, a.indoor_archive))))
    write_json(a.output, dict(panels=panels, script_sha256=sha256(__file__),
        search_sha256=sha256("scripts/research_balloon_search.py"),
        inputs={name: sha256(path) for name, path in dict(manifest=a.manifest,
            original_archive=a.original_archive, indoor_archive=a.indoor_archive,
            queue=a.queue / "review.json", review=a.review).items()},
        scope="Training-positive per-target coverage; incomplete labels, no precision claim",
        classifier_evaluated=False, test_evaluated=False, deployment_approved=False, pi_timing_measured=False))
    print({panel: {v: dict(counts=r["counts"], mean_candidates=r["mean_candidates"],
                          desktop_mean_search_ms=r["desktop_mean_search_ms"]) for v, r in result.items()}
           for panel, result in panels.items()})
