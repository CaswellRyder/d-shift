"""Visually reviewed blue positives with exact source, license and pixel guards."""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from dtr.data import read_json, sha256
from scripts.acquire_blue_balloon_candidates import OUTPUT as ROOT
from scripts.prepare_blue_balloon_candidates import OUTPUT as QUEUE
from scripts.prepare_openimages_balloon_review import checked_path
from scripts.openimages_balloon_training import review_labels

REVIEW = Path('configs/blue-balloon-candidates-review-20261011.json')


def checked_extra():
    review, queue = read_json(REVIEW), read_json(QUEUE / 'review.json')
    acquisition, licenses = read_json(ROOT / 'acquisition.json'), read_json(ROOT / 'license-verification.json')
    if (review['queue'] != str(QUEUE) or review['queue_sha256'] != sha256(QUEUE / 'review.json')
            or review['reviewed_sheets'] != queue['sheets']
            or queue['script_sha256'] != sha256('scripts/prepare_blue_balloon_candidates.py')
            or queue['acquisition_sha256'] != sha256(ROOT / 'acquisition.json')
            or queue['license_sha256'] != sha256(ROOT / 'license-verification.json')
            or licenses['acquisition_sha256'] != sha256(ROOT / 'acquisition.json')
            or licenses['script_sha256'] != sha256('scripts/verify_openimages_licenses.py')
            or acquisition['script_sha256'] != sha256('scripts/acquire_blue_balloon_candidates.py')):
        raise ValueError('Review/source identity changed')
    checks = {**queue['reference_inputs'], **queue['helpers_sha256'], **acquisition['helpers_sha256']}
    for key, name in (('selection_sha256', 'selection.json'), ('attribution_sha256', 'attribution.json')):
        checks[str(ROOT / name)] = acquisition[key]
    if any(sha256(p) != h for p, h in checks.items()):
        raise ValueError('Source helpers, references or metadata changed')
    for name, h in queue['sheets'].items():
        checked_path(QUEUE, name, h)
    labels = review_labels(review, len(queue['samples']))
    if len(labels) != 7 or set(labels.values()) != {'blue_balloon'}:
        raise ValueError('Require7 reviewed blue-positive crops')
    verified = {r['image_id'] for r in licenses['results'] if r['verified'] is True}
    sources = {r['image_id']: r for r in acquisition['images']}
    rows = []
    for i, label in sorted(labels.items()):
        r = queue['samples'][i]
        f = queue['frames'][r['frame_id']]
        source = sources[f['image_id']]
        if (r['id'] != i or f['id'] != r['frame_id'] or r['split'] != 'train'
                or r['source_image'] != 'openimages/train/' + f['image_id']
                or f['image_id'] not in verified or f['attribution'] != source['attribution']
                or f['attribution']['Subset'] != 'train' or f['original_sha256'] != source['sha256']
                or f['original_path'] != source['path'] or f['reference_hamming'] <= 8
                or (f['selected_hamming'] is not None and f['selected_hamming'] <= 8)):
            raise ValueError('Unsafe or changed source/crop')
        with Image.open(checked_path(ROOT, f['original_path'], f['original_sha256'])) as im:
            original = np.asarray(im.convert('RGB'))
        with Image.open(checked_path(QUEUE, f['path'], f['sha256'])) as im:
            scan = np.asarray(im.convert('RGB'))
        if not np.array_equal(scan, cv2.resize(original, (320, 240), interpolation=cv2.INTER_AREA)):
            raise ValueError('Frame differs from source')
        path = checked_path(QUEUE, r['path'], r['sha256'])
        a, b, c, d = r['crop_box']
        if not (0 <= a < c <= 320 and 0 <= b < d <= 240):
            raise ValueError('Invalid reviewed crop bounds')
        with Image.open(path) as im:
            if not np.array_equal(np.asarray(im.convert('RGB')), scan[b:d, a:c]):
                raise ValueError('Crop differs from reviewed frame')
        rows.append(dict(r, label=label, file=str(path), preserve=False, view='original', attribution=f['attribution']))
    if len({r['sha256'] for r in rows}) != len(rows) or len({r['source_image'] for r in rows}) != 5:
        raise ValueError('Duplicate pixels or changed photograph count')
    return rows
