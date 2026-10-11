"""Training-only native-detail search audit; never upscale and call it new detail."""
import argparse
import hashlib
import io
from pathlib import Path
import time
import zipfile

import cv2
import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, write_json
from scripts.audit_public_balloon_export import bounded_read
from scripts.audit_balloon_search_stages import original_frames, indoor_frames, training_source
from scripts.evaluate_balloon_edge_search import BASE, ORIGINAL, INDOOR, TRAIN_QUEUES, coverage
from scripts.research_balloon_search import candidate, experimental, select

MANIFEST = 'data/balloon-red-blue-expansion2-admit-20261011/manifest.json'


def high_proposals(rgb):
    """Double processing resolution, same physical gates and twelve-crop cap.

    Return 320x240 coordinates. MSER point counts scale with pixel area; geometry
    and proposal ranking are measured back in baseline coordinates.
    """
    if rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8:
        raise ValueError('Require 640x480 RGB uint8')
    pool = []
    planes = (cv2.subtract(rgb[:, :, 0], cv2.max(rgb[:, :, 1], rgb[:, :, 2])),
              cv2.subtract(rgb[:, :, 2], cv2.max(rgb[:, :, 0], rgb[:, :, 1])))
    for group, plane in enumerate(planes):
        regions, _ = cv2.MSER_create(5, 24, 100000).detectRegions(plane)
        for points in regions:
            if float(plane[points[:, 1], points[:, 0]].mean()) < 15:
                continue
            result = candidate(points.astype(np.float32)/2., group, (240, 320, 3))
            if result:
                pool.append(result)
    return select(pool)


def native_frame(archive, source, expected, allowed):
    training_source(source)
    member = source if source in allowed else 'train/'+source
    if member not in allowed:
        raise ValueError('Native frame is not an admitted training source')
    with Image.open(io.BytesIO(bounded_read(archive, member, 30_000_000))) as image:
        pixels = np.asarray(image.convert('RGB'))
    if hashlib.sha256(pixels.tobytes()).hexdigest() != allowed[member]:
        raise ValueError('Native source pixels changed')
    if not np.array_equal(cv2.resize(pixels, (320, 240), interpolation=cv2.INTER_AREA), expected):
        raise ValueError('Native source does not reproduce reviewed baseline')
    return pixels


def training():
    manifest = read_json(MANIFEST)
    allowed = {r['source_image']: r['source_sha256'] for r in manifest['samples'] if r['split'] == 'train'}
    forbidden = {r['source_sha256'] for r in manifest['samples'] if r['split'] != 'train'}
    if any(v in forbidden for v in allowed.values()):
        raise ValueError('Cross-split source identity')
    panels = {'original': (ORIGINAL, original_frames(BASE, ORIGINAL))}
    panels.update({name: (INDOOR, indoor_frames('data/'+name, 'configs/'+name+'.json', BASE, INDOOR))
                   for name in TRAIN_QUEUES})
    results = {}
    for name, (archive_path, frames) in panels.items():
        rows = []
        with zipfile.ZipFile(archive_path) as archive:
            for source, rgb, truth in frames:
                native = native_frame(archive, source, rgb, allowed)
                has_detail = native.shape[0] >= 480 and native.shape[1] >= 640
                start = time.perf_counter()
                base = experimental(rgb, 'mser')
                baseline_ms = (time.perf_counter()-start)*1000
                start = time.perf_counter()
                high = high_proposals(cv2.resize(native, (640, 480), interpolation=cv2.INTER_AREA)) if has_detail else base
                high_ms = (time.perf_counter()-start)*1000 if has_detail else None
                if len(high) > 12:
                    raise ValueError('Neural-call budget changed')
                rows.append(dict(source=source, native_shape=list(native.shape), native_detail_used=has_detail,
                    truth=truth, baseline=coverage(truth, base), high=coverage(truth, high),
                    baseline_proposals=base, high_proposals=high,
                    baseline_host_search_ms=baseline_ms, high_host_search_ms=high_ms))
        totals = {label: dict(targets=sum(r['baseline'][label]['targets'] for r in rows),
            baseline=sum(r['baseline'][label]['localized'] for r in rows),
            high=sum(r['high'][label]['localized'] for r in rows)) for label in ('red_balloon', 'blue_balloon')}
        results[name] = dict(frames=rows, totals=totals)
    return dict(panels=results, script_sha256=sha256(__file__),
        baseline_script_sha256=sha256('scripts/research_balloon_search.py'),
        inputs={p: sha256(p) for p in (MANIFEST, BASE, ORIGINAL, INDOOR)},
        training_reviews={q: dict(queue_sha256=sha256('data/'+q+'/review.json'),
            review_sha256=sha256('configs/'+q+'.json')) for q in TRAIN_QUEUES},
        scope='TRAIN-only one-to-one proposal coverage; incomplete scene truth, no precision or classifier claim',
        caveat='Search timing includes high-resolution resize but not camera capture. macOS timings are not Pi predictions.',
        classifier_evaluated=False, training_performed=False, test_evaluated=False,
        pi_tested=False, deployment_approved=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = training()
    write_json(args.output, report)
    print({k: v['totals'] for k, v in report['panels'].items()})
