"""Source-reproduced, visually reviewed TRAIN-only additions; never relabel test images."""
import hashlib
import io
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np
from PIL import Image

from dtr.data import read_json, sha256
from scripts.balloon_expansion3 import partition, BOOTSTRAP
from scripts.build_indoor_balloon_training import admitted
from scripts.prepare_indoor_balloon_review import ARCHIVE_SHA, MAP, family
from scripts.audit_public_balloon_export import source_group, bounded_read
from scripts.prepare_openimages_balloon_review import checked_path
from scripts.screen_indoor_expansion5 import QUEUE, OUTPUT as SCREEN

REVIEW = Path('configs/engdes2-indoor-expansion5-review-20261011.json')
ARCHIVE = Path('data/raw/engdes2-red-blue-v1-20261008/dataset.zip')
PRIOR = ('engdes2-indoor-review-20261008', 'engdes2-indoor-expansion-review-20261008',
         'engdes2-indoor-expansion2-review-20261011', 'engdes2-indoor-expansion3-review-20261011',
         'engdes2-indoor-expansion4-review-20261011')


def checked_extra():
    queue, review, screen = read_json(QUEUE / 'review.json'), read_json(REVIEW), read_json(SCREEN)
    partition(review, len(queue['samples']))
    if (sha256(ARCHIVE) != ARCHIVE_SHA or queue['archive_sha256'] != ARCHIVE_SHA
            or queue['script_sha256'] != sha256('scripts/prepare_indoor_balloon_review.py')
            or queue['excluded_queue_sha256s'] != [sha256(f'data/{q}/review.json') for q in PRIOR]
            or screen['queue_sha256'] != sha256(QUEUE / 'review.json')
            or screen['script_sha256'] != sha256('scripts/screen_indoor_expansion5.py')
            or len(screen['frames']) != len(queue['frames'])):
        raise ValueError('Quarantine provenance changed')
    for path, digest in {**screen['reference_inputs'], **screen['helpers_sha256']}.items():
        if sha256(path) != digest:
            raise ValueError('Duplicate-screen references changed')
    for path, digest in review['reviewed_sheets'].items():
        checked_path(QUEUE, path, digest)
    rows = admitted(QUEUE, REVIEW, BOOTSTRAP)
    previous = {source_group(f['source']) for name in PRIOR
                for f in read_json(f'data/{name}/review.json')['frames']}
    images = {}
    with zipfile.ZipFile(ARCHIVE) as archive:
        splits = {s: json.loads(bounded_read(archive, f'{s}/_annotations.coco.json', 50_000_000))
                  for s in ('train', 'valid', 'test')}
        forbidden = {source_group(r['file_name']) for s in ('valid', 'test') for r in splits[s]['images']}
        annotations = {r['id']: r for r in splits['train']['annotations']}
        categories = {r['id']: r['name'] for r in splits['train']['categories']}
        sources = {r['file_name']: r for r in splits['train']['images']}
        for row in rows:
            frame = queue['frames'][row['frame_id']]
            check = screen['frames'][row['frame_id']]
            if (frame['id'] != row['frame_id'] or check['id'] != frame['id'] or check['sha256'] != frame['sha256']
                    or check['source'] != frame['source'] or check['eligible'] is not True
                    or check['reference_hamming'] <= 8 or check['earlier_selected_hamming'] is not None and check['earlier_selected_hamming'] <= 8
                    or family(frame['source']) is None or source_group(frame['source']) in forbidden | previous
                    or row['source_image'] != 'train/' + frame['source']):
                raise ValueError('Reserved, repeated or changed training source')
            if frame['id'] not in images:
                with Image.open(io.BytesIO(bounded_read(archive, row['source_image'], 20_000_000))) as im:
                    original = np.asarray(im.convert('RGB'))
                expected = cv2.resize(original, (320, 240), interpolation=cv2.INTER_AREA)
                with Image.open(checked_path(QUEUE, frame['path'], frame['sha256'])) as im:
                    if not np.array_equal(np.asarray(im.convert('RGB')), expected):
                        raise ValueError('Frame pixels do not reproduce source')
                images[frame['id']] = expected, hashlib.sha256(original.tobytes()).hexdigest()
            pixels, digest = images[frame['id']]
            if digest != row['source_sha256']:
                raise ValueError('Source pixel identity changed')
            if row['label'] != 'background':
                ann = annotations[row['annotation_id']]
                if (ann['image_id'] != sources[frame['source']]['id']
                        or MAP[categories[ann['category_id']]] != row['label']):
                    raise ValueError('Annotation identity or color changed')
            a, b, c, d = row['crop_box']
            if not 0 <= a < c <= 320 or not 0 <= b < d <= 240:
                raise ValueError('Crop bounds changed')
            path = checked_path(QUEUE, row['path'], row['sha256'])
            with Image.open(path) as im:
                if not np.array_equal(np.asarray(im.convert('RGB')), pixels[b:d, a:c]):
                    raise ValueError('Crop does not match reviewed frame')
    return [dict(r, file=str(QUEUE / r['path']), preserve=False, view='original') for r in rows]
