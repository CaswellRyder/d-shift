"""Fail-closed admission of visually reviewed current-model TRAIN mistakes."""
from pathlib import Path

import numpy as np
from PIL import Image

from dtr.data import read_json, sha256
from scripts.prepare_current_color_clutter import OUTPUT as QUEUE
from scripts.prepare_blue_false_alarm_review import source_frames
from scripts.prepare_grid_clutter_review import guarded_background
from scripts.prepare_openimages_balloon_review import checked_path
from scripts.openimages_balloon_training import review_labels
from scripts.blue_false_alarm_data import data as base_data

REVIEW = Path('configs/current-color-clutter-review-20261011.json')


def checked_extra():
    review, queue = read_json(REVIEW), read_json(QUEUE/'review.json')
    if (review['queue'] != str(QUEUE) or review['queue_sha256'] != sha256(QUEUE/'review.json')
            or queue['script_sha256'] != sha256('scripts/prepare_current_color_clutter.py')
            or review['reviewed_sheets'] != queue['sheets'] or queue['guard_pixels'] != 16
            or any(sha256(p) != h for p,h in {**queue['helpers_sha256'],**queue['previous_queues']}.items())):
        raise ValueError('Frozen review changed')
    for name,digest in queue['sheets'].items():
        checked_path(QUEUE,name,digest)
    for identity in [queue['classifier'],*queue['models'].values()]:
        path = Path(identity['path'])
        if sha256(path) != identity['sha256'] or sha256(path.with_suffix('.json')) != identity['metadata_sha256']:
            raise ValueError('Mining model changed')
    sources = {r['source']:(r,pixels) for r,pixels in source_frames()}
    labels = review_labels(review,len(queue['samples']))
    if len(labels) != 15 or set(labels.values()) != {'background'}:
        raise ValueError('Require15 reviewed backgrounds')
    selected = []
    for i,label in sorted(labels.items()):
        row = queue['samples'][i]
        frame = queue['frames'][row['frame_id']]
        source,pixels = sources[row['source']]
        if (row['id'] != i or row['split'] != 'train' or row['source'] in queue['excluded_sources']
                or any(frame[k] != v for k,v in source.items())
                or not guarded_background(row['crop_box'],[{'box':b} for b in source['boxes']])):
            raise ValueError('Unsafe or altered training source')
        a,b,c,d = row['crop_box']
        with Image.open(checked_path(QUEUE,frame['path'],frame['sha256'])) as im:
            if not np.array_equal(np.asarray(im.convert('RGB')),pixels):
                raise ValueError('Source pixels changed')
        path = checked_path(QUEUE,row['path'],row['sha256'])
        with Image.open(path) as im:
            if not np.array_equal(np.asarray(im.convert('RGB')),pixels[b:d,a:c]):
                raise ValueError('Crop differs from reviewed pixels')
        selected.append(dict(row,label=label,file=str(path),source_image=row['source'],view='original',preserve=False))
    return selected


def data(mode):
    if mode not in ('control','admit'):
        raise ValueError('Require paired arm')
    rows,blue,clutter,real = base_data('admit')
    extra = checked_extra()
    known = {r['sha256']:r['label'] for r in rows}
    if len({r['sha256'] for r in extra}) != len(extra) or any(r['sha256'] in known for r in extra):
        raise ValueError('New sample duplicates existing training data')
    pool = [r for r in rows if r['label'] == 'background' and r['preserve']]
    indices = np.random.default_rng(47).choice(len(pool),len(extra),replace=False)
    chosen = extra if mode == 'admit' else [pool[int(i)] for i in indices]
    additions = [dict(r,preserve=False,view='original') for _ in range(2) for r in chosen]
    return rows+additions,blue,clutter,real,extra
