"""Load a sealed, explicitly reviewed real-photo batch, never inferred labels."""
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from dtr.data import read_json, sha256
from scripts.openimages_balloon_training import review_labels
from scripts.prepare_openimages_balloon_review import checked_path


def inventory(rows, review):
    if (len(rows) != review['admitted_crops']
            or len({r['sha256'] for r in rows}) != len(rows)
            or len({r['source_image'] for r in rows}) != review['admitted_photographs']
            or Counter(r['label'] for r in rows) != review['label_counts']
            or max(Counter(r['source_image'] for r in rows).values(), default=0) > review['max_crops_per_photo']):
        raise ValueError('Reviewed crop inventory or source balance changed')


def load_batch(review_path, expected_sha256):
    if sha256(review_path) != expected_sha256:
        raise ValueError('Sealed review changed')
    review = read_json(review_path)
    root, output = Path(review['root']), Path(review['queue'])
    queue = read_json(output/'review.json')
    acquisition, licenses = read_json(root/'acquisition.json'), read_json(root/'license-verification.json')
    if (review['test_evaluated'] is not False or review['queue_sha256'] != sha256(output/'review.json')
            or queue['root'] != str(root) or queue['sheets'] != review['reviewed_sheets']
            or queue['script_sha256'] != sha256('scripts/prepare_blue_balloon_batch.py')
            or queue['acquisition_sha256'] != sha256(root/'acquisition.json')
            or queue['license_sha256'] != sha256(root/'license-verification.json')
            or licenses['acquisition_sha256'] != sha256(root/'acquisition.json')
            or licenses['script_sha256'] != sha256('scripts/verify_openimages_licenses.py')
            or acquisition['script_sha256'] != sha256('scripts/acquire_blue_balloon_batch.py')):
        raise ValueError('Acquisition, review or license identity changed')
    checks = {**queue['reference_inputs'], **queue['helpers_sha256'], **acquisition['helpers_sha256']}
    for key, name in (('selection_sha256', 'selection.json'), ('attribution_sha256', 'attribution.json')):
        checks[str(root/name)] = acquisition[key]
    if any(sha256(p) != h for p, h in checks.items()):
        raise ValueError('Source or duplicate reference changed')
    for path, digest in review['reviewed_sheets'].items():
        checked_path(output, path, digest)
    labels = review_labels(review, len(queue['samples']))
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
            raise ValueError('Unsafe or changed source')
        with Image.open(checked_path(root, frame['original_path'], frame['original_sha256'])) as im:
            rgb = np.asarray(im.convert('RGB'))
        with Image.open(checked_path(output, frame['path'], frame['sha256'])) as im:
            scan = np.asarray(im.convert('RGB'))
        if not np.array_equal(scan, cv2.resize(rgb, (320, 240), interpolation=cv2.INTER_AREA)):
            raise ValueError('Review frame differs from source')
        a, b, c, d = row['crop_box']
        if not (0 <= a < c <= 320 and 0 <= b < d <= 240):
            raise ValueError('Invalid crop bounds')
        path = checked_path(output, row['path'], row['sha256'])
        with Image.open(path) as im:
            if not np.array_equal(np.asarray(im.convert('RGB')), scan[b:d, a:c]):
                raise ValueError('Crop differs from reviewed pixels')
        rows.append(dict(row, label=label, file=str(path), preserve=False, view='original',
            source_sha256=frame['original_sha256'], attribution=frame['attribution']))
    inventory(rows, review)
    return rows
