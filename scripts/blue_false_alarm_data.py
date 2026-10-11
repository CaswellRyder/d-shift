"""Verified real-blue negatives, paired with count-matched existing TRAIN backgrounds."""
from pathlib import Path

import numpy as np
from PIL import Image

from dtr.data import read_json, sha256
from scripts.openimages_balloon_training import review_labels
from scripts.openimages_blue_expansion import data as blue_data
from scripts.prepare_blue_false_alarm_review import source_frames
from scripts.prepare_grid_clutter_review import guarded_background
from scripts.prepare_openimages_balloon_review import checked_path

REVIEW = Path('configs/blue-false-alarm-review-20261011.json')
REVIEW_SHA = '240d86017ce93782128359b44ba7784b52c0a0c2c2f31fb4a3ad6f4a9b2d5b19'
QUEUE = Path('data/blue-false-alarm-review-20261011')


def checked_extra():
    if sha256(REVIEW) != REVIEW_SHA:
        raise ValueError('Frozen visual review changed')
    review, queue = read_json(REVIEW), read_json(QUEUE / 'review.json')
    if (sha256(QUEUE / 'review.json') != review['queue_sha256']
            or review['queue'] != str(QUEUE) or review['reviewed_sheets'] != queue['sheets']
            or queue['script_sha256'] != sha256('scripts/prepare_blue_false_alarm_review.py')
            or queue['guard_pixels'] != 16):
        raise ValueError('Source queue or mining policy changed')
    for name, digest in queue['sheets'].items():
        checked_path(QUEUE, name, digest)
    for identity in [queue['classifier'], *queue['models'].values()]:
        path = Path(identity['path'])
        if sha256(path) != identity['sha256'] or sha256(path.with_suffix('.json')) != identity['metadata_sha256']:
            raise ValueError('Mining model identity changed')
    # This rechecks all source pixels, FIT folds, complete-scene review, public
    # licenses and duplicate-screen receipts. No development pixels are read.
    sources = {row['source']: (row, pixels) for row, pixels in source_frames()}
    labels = review_labels(review, len(queue['samples']))
    if len(labels) != 39 or set(labels.values()) != {'background'}:
        raise ValueError('Require39 explicitly reviewed backgrounds')
    selected = []
    for i, label in sorted(labels.items()):
        row = queue['samples'][i]
        frame = queue['frames'][row['frame_id']]
        source, pixels = sources[row['source']]
        if (row['id'] != i or row['split'] != 'train'
                or any(frame[k] != v for k, v in source.items())
                or not guarded_background(row['crop_box'], [{'box': b} for b in source['boxes']])):
            raise ValueError('Unsafe TRAIN crop identity')
        with Image.open(checked_path(QUEUE, frame['path'], frame['sha256'])) as im:
            if not np.array_equal(np.asarray(im.convert('RGB')), pixels):
                raise ValueError('Frame differs from source')
        path = checked_path(QUEUE, row['path'], row['sha256'])
        a, b, c, d = row['crop_box']
        with Image.open(path) as im:
            if not np.array_equal(np.asarray(im.convert('RGB')), pixels[b:d, a:c]):
                raise ValueError('Crop differs from reviewed source pixels')
        selected.append(dict(row, label=label, source_image=row['source'], file=str(path),
            view='original', preserve=False, source_sha256=source['source_sha256']))
    return selected


def data(mode):
    if mode not in ('control', 'admit'):
        raise ValueError('Require paired arm')
    rows, blue, clutter = blue_data('admit')
    # Newly learned blue positives are now protected alongside all ordinary
    # positives. Prior hard backgrounds remain unprotected hard CE rows.
    rows = [dict(r, preserve=r['preserve'] or r['label'] != 'background') for r in rows]
    extra = checked_extra()
    known = {r['sha256']: r['label'] for r in rows}
    if (len({r['sha256'] for r in extra}) != len(extra)
            or any(r['sha256'] in known and known[r['sha256']] != 'background' for r in extra)):
        raise ValueError('Duplicate new crop or conflict with known positive')
    pool = [r for r in rows if r['label'] == 'background' and r['preserve']]
    indices = np.random.default_rng(47).choice(len(pool), len(extra), replace=False)
    chosen = extra if mode == 'admit' else [pool[int(i)] for i in indices]
    additions = [dict(r, preserve=False, view='original') for _ in range(2) for r in chosen]
    return rows + additions, blue, clutter, extra
