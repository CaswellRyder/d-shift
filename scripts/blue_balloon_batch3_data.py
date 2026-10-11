"""Verify seven explicit real TRAIN crop labels without altering prior data."""
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from dtr.data import read_json, sha256
from scripts.openimages_balloon_training import review_labels
from scripts.prepare_openimages_balloon_review import checked_path

ROOT = Path('data/raw/openimages-blue-batch3-20261011')
QUEUE = Path('data/blue-balloon-batch3-review-20261011')
REVIEW = Path('configs/blue-balloon-batch3-review-20261011.json')
EXPECTED = {2: 'blue_balloon', 7: 'blue_balloon', 8: 'red_balloon', 9: 'blue_balloon',
            19: 'blue_balloon', 21: 'blue_balloon', 24: 'blue_balloon'}


def checked_labels(review, count):
    labels = review_labels(review, count)
    if count != 28 or labels != EXPECTED or review.get('test_evaluated') is not False:
        raise ValueError('Explicit batch3 crop admission changed')
    return labels


def checked_extra():
    review, queue = read_json(REVIEW), read_json(QUEUE/'review.json')
    acquisition, licenses = read_json(ROOT/'acquisition.json'), read_json(ROOT/'license-verification.json')
    if (review['queue'] != str(QUEUE) or review['queue_sha256'] != sha256(QUEUE/'review.json')
            or review['reviewed_sheets'] != queue['sheets'] or queue['root'] != str(ROOT)
            or queue['script_sha256'] != sha256('scripts/prepare_blue_balloon_batch.py')
            or queue['acquisition_sha256'] != sha256(ROOT/'acquisition.json')
            or queue['license_sha256'] != sha256(ROOT/'license-verification.json')
            or licenses['acquisition_sha256'] != sha256(ROOT/'acquisition.json')
            or licenses['script_sha256'] != sha256('scripts/verify_openimages_licenses.py')
            or acquisition['script_sha256'] != sha256('scripts/acquire_blue_balloon_batch.py')):
        raise ValueError('Changed review, acquisition or license identity')
    checks = {**queue['reference_inputs'], **queue['helpers_sha256'], **acquisition['helpers_sha256']}
    for key, name in (('selection_sha256', 'selection.json'), ('attribution_sha256', 'attribution.json')):
        checks[str(ROOT/name)] = acquisition[key]
    if any(sha256(path) != digest for path, digest in checks.items()):
        raise ValueError('Changed source metadata or duplicate references')
    for name, digest in queue['sheets'].items():
        checked_path(QUEUE, name, digest)
    labels = checked_labels(review, len(queue['samples']))
    verified = {r['image_id'] for r in licenses['results'] if r['verified'] is True}
    sources = {r['image_id']: r for r in acquisition['images']}
    rows = []
    for i, label in sorted(labels.items()):
        row = queue['samples'][i]
        frame = queue['frames'][row['frame_id']]
        source = sources[frame['image_id']]
        if (row['id'] != i or row['split'] != 'train' or frame['id'] != row['frame_id']
                or row['source_image'] != 'openimages/train/'+frame['image_id']
                or frame['image_id'] not in verified or frame['attribution'] != source['attribution']
                or frame['attribution']['Subset'] != 'train' or frame['original_sha256'] != source['sha256']
                or frame['original_path'] != source['path'] or frame['reference_hamming'] <= 8
                or (frame['selected_hamming'] is not None and frame['selected_hamming'] <= 8)):
            raise ValueError('Unsafe source or changed reviewed crop')
        with Image.open(checked_path(ROOT, frame['original_path'], frame['original_sha256'])) as im:
            original = np.asarray(im.convert('RGB'))
        with Image.open(checked_path(QUEUE, frame['path'], frame['sha256'])) as im:
            scan = np.asarray(im.convert('RGB'))
        if not np.array_equal(scan, cv2.resize(original, (320, 240), interpolation=cv2.INTER_AREA)):
            raise ValueError('Review frame differs from original')
        a, b, c, d = row['crop_box']
        if not (0 <= a < c <= 320 and 0 <= b < d <= 240):
            raise ValueError('Invalid crop bounds')
        path = checked_path(QUEUE, row['path'], row['sha256'])
        with Image.open(path) as im:
            if not np.array_equal(np.asarray(im.convert('RGB')), scan[b:d, a:c]):
                raise ValueError('Crop differs from reviewed pixels')
        rows.append(dict(row, label=label, file=str(path), preserve=False, view='original',
                         source_sha256=frame['original_sha256'], attribution=frame['attribution']))
    if (len({r['sha256'] for r in rows}) != 7 or len({r['source_image'] for r in rows}) != 4
            or Counter(r['label'] for r in rows) != {'blue_balloon': 6, 'red_balloon': 1}):
        raise ValueError('Changed scene/crop inventory')
    return rows
