"""Explicit visual labels and verified source pages for a TRAIN-only trial."""
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from dtr.data import read_json, sha256
from scripts.prepare_openimages_balloon_review import checked_path
from scripts.train_balloon_fragments import checked_data, MANIFEST, QUEUE as FRAGMENT_QUEUE
from scripts.mine_red_blue_negatives import safe_negative

QUEUE = 'data/openimages-balloon-review-20261011'
REVIEW = 'configs/openimages-balloon-review-20261011.json'
ROOT = Path('data/raw/openimages-balloon-train-20261011')
LICENSE = ROOT/'license-verification.json'
QUEUE_SHA = '4bd029db3534597c2787ed8a559729a2912c213cf114eceb7a7ef62d3936a221'
REVIEW_SHA = '51255d208e861b0ce1af002a4bd0cd1376646132aae1224c5c49eff2092876dc'
LICENSE_SHA = '9b77f2dfa36267ccc98d99b2996ab3f50437a1699697d3d2fc41426055ce4367'


def safe_background(row, frame):
    return (row['label'] != 'background' or row['kind'] != 'unmatched_proposal'
            or safe_negative(row['crop_box'],frame['boxes']))


def review_labels(review, count):
    if (review.get('training_approved') is not True or review.get('split') != 'train'
            or review.get('deployment_approved') is not False
            or set(review['labels_by_class']) != {'background','red_balloon','blue_balloon'}):
        raise ValueError('Require explicit training-only labels')
    labeled = [(i,label) for label,ids in review['labels_by_class'].items() for i in ids]
    ids = [i for i,_ in labeled]+review['exclude']
    if any(type(i) is not int for i in ids) or len(ids) != len(set(ids)) or set(ids) != set(range(count)):
        raise ValueError('Incomplete or overlapping review')
    return dict(labeled)


def checked_extra():
    root = Path(QUEUE)
    if sha256(root/'review.json') != QUEUE_SHA or sha256(REVIEW) != REVIEW_SHA or sha256(LICENSE) != LICENSE_SHA:
        raise ValueError('Frozen source review changed')
    queue,review,licenses = read_json(root/'review.json'),read_json(REVIEW),read_json(LICENSE)
    if (queue['acquisition_sha256'] != sha256(ROOT/'acquisition.json')
            or licenses['acquisition_sha256'] != queue['acquisition_sha256']):
        raise ValueError('Acquisition identity changed')
    for path,digest in queue['reference_inputs'].items():
        if sha256(path) != digest:
            raise ValueError('Duplicate reference identity changed')
    labels = review_labels(review,len(queue['samples']))
    verified = {r['image_id'] for r in licenses['results'] if r['verified'] is True}
    frames, pixels = {}, {}
    for f in queue['frames']:
        if (f['reference_hamming'] <= 8 or (f['selected_hamming'] is not None and f['selected_hamming'] <= 8)
                or f['id'] in frames or f['attribution']['Subset'] != 'train'):
            raise ValueError('Invalid or duplicate training frame')
        path = checked_path(ROOT,f['original_path'],f['original_sha256'])
        with Image.open(path) as im:
            original = np.asarray(im.convert('RGB'))
        with Image.open(checked_path(root,f['path'],f['sha256'])) as im:
            scan = np.asarray(im.convert('RGB'))
        if not np.array_equal(scan,cv2.resize(original,(320,240),interpolation=cv2.INTER_AREA)):
            raise ValueError('Frame differs from source pixels')
        frames[f['id']],pixels[f['id']] = f,scan
    eligible = []
    for i,label in sorted(labels.items()):
        row = queue['samples'][i]
        frame = frames[row['frame_id']]
        if row['id'] != i or row['split'] != 'train' or row['source_image'] != 'openimages/train/'+frame['image_id']:
            raise ValueError('Invalid crop identity')
        if frame['image_id'] not in verified:
            continue
        a,b,c,d = row['crop_box']
        with Image.open(checked_path(root,row['path'],row['sha256'])) as im:
            crop = np.asarray(im.convert('RGB'))
        if not np.array_equal(crop,pixels[row['frame_id']][b:d,a:c]):
            raise ValueError('Crop differs from reviewed frame')
        labeled = dict(row,label=label)
        if safe_background(labeled,frame):
            eligible.append(labeled)
    # All reviewed positives; at most two backgrounds per scene, 36 total.
    # Hash order is independent of current classifier scores or held-out results.
    positives = [r for r in eligible if r['label'] != 'background']
    backgrounds,per_frame = [],Counter()
    for r in sorted(eligible,key=lambda r:r['sha256']):
        if r['label']=='background' and per_frame[r['frame_id']] < 2 and len(backgrounds)<36:
            backgrounds.append(r)
            per_frame[r['frame_id']] += 1
    return sorted(positives+backgrounds,key=lambda r:r['id'])


def data(mode):
    if mode not in ('control','admit'):
        raise ValueError('Require matched control or admit')
    ordinary,fragments = checked_data()
    extra = checked_extra()
    rows = [dict(r,file=str(Path(MANIFEST).parent/r['path']),preserve=True) for r in ordinary]
    rows += [dict(r,label='blue_balloon',file=str(Path(FRAGMENT_QUEUE)/r['path']),preserve=True)
             for r in fragments]*4
    pools = {label:[r for r in rows if r['label']==label] for label in ('background','red_balloon','blue_balloon')}
    rng = np.random.default_rng(47)
    additions = []
    for _ in range(2):
        for r in extra:
            chosen = (dict(r,file=str(Path(QUEUE)/r['path'])) if mode=='admit'
                      else pools[r['label']][int(rng.integers(len(pools[r['label']])) )])
            additions.append(dict(chosen,preserve=False))
    return rows+additions,extra
