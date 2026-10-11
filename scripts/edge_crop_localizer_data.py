"""Real TRAIN scene crops with clipped geometry and unchanged source boundaries."""
import hashlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from dtr.data import read_json, sha256
from scripts.balloon_dense_complete import encode_complete, validate_derivative
from scripts.blue_scene_localizer_data import data as original_data, dependencies
from scripts.prepare_grid_clutter_review import BASE, BASE_SHA
from scripts.train_balloon_grid_convergence import source_data

PROTOCOL = Path('configs/edge-crop-localizer-20261011.json')
WINDOWS = ((0, 0, 240, 180), (80, 0, 320, 180), (0, 60, 240, 240),
           (80, 60, 320, 240), (40, 30, 280, 210))


def tensor_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def crop_scene(rgb, truth, window):
    if rgb.shape != (240, 320, 3) or rgb.dtype != np.uint8 or tuple(window) not in WINDOWS:
        raise ValueError('Require canonical pixels and fixed crop window')
    x, y, right, bottom = window
    sx, sy = 320/(right-x), 240/(bottom-y)
    transformed = []
    for row in truth:
        a, b, c, d = row['box']
        if (row['label'] not in ('red_balloon', 'blue_balloon')
                or not np.isfinite([a, b, c, d]).all() or not 0 <= a < c <= 320 or not 0 <= b < d <= 240):
            raise ValueError('Invalid source annotation')
        a, b, c, d = max(a, x), max(b, y), min(c, right), min(d, bottom)
        if a < c and b < d:
            transformed.append(dict(row, box=[(a-x)*sx, (b-y)*sy, (c-x)*sx, (d-y)*sy]))
    image = cv2.resize(rgb[y:bottom, x:right], (320, 240), interpolation=cv2.INTER_LINEAR)
    target, collisions = encode_complete(transformed, [])
    return image, transformed, target, collisions


def inputs(arm):
    if arm not in ('control', 'edge_crop'):
        raise ValueError('Unknown paired augmentation arm')
    p = read_json(PROTOCOL)
    if tuple(map(tuple, p['windows'])) != WINDOWS or sha256(BASE/'manifest.json') != BASE_SHA:
        raise ValueError('Crop protocol/source changed')
    doc, base_x, base_y, base_folds, groups = source_data('coarse')
    x, y, folds, extra, original_groups = original_data('background')
    if (groups != original_groups or not np.array_equal(x[:129], base_x)
            or not np.array_equal(y[:129], base_y) or not np.array_equal(folds[:129], base_folds)):
        raise ValueError('Original training view changed')
    complete_doc = read_json(BASE/'manifest.json')
    complete_rows = [complete_doc['frames'][i] for i in sorted(validate_derivative(complete_doc))]
    original_index = {r['source']: i for i, r in enumerate(doc['frames'])}
    complete = [original_index[r['source']] for r in complete_rows]
    if len(complete) != 64 or any(folds[i] == 0 for i in complete):
        raise ValueError('Only complete FIT frames may be cropped')
    added_x, added_y, records, clipped = [], [], [], 0
    for index, source_row in zip(complete, complete_rows, strict=True):
        row = doc['frames'][index]
        if any(row[k] != source_row[k] for k in ('source', 'source_sha256', 'sha256', 'truth')):
            raise ValueError('Expanded source differs from complete review')
        path = BASE/source_row['path']
        if 'IMG_' in row['source'] or sha256(path) != row['sha256']:
            raise ValueError('Forbidden or changed source')
        with Image.open(path) as im:
            rgb = np.asarray(im.convert('RGB'))
        for window in WINDOWS:
            image, truth, target, collisions = crop_scene(rgb, row['truth'], window)
            before = {t['annotation_id']: t for t in row['truth']}
            clipped += sum(any(t['box'][j] == v for j, v in ((0, 0), (1, 0), (2, 320), (3, 240)))
                           for t in truth)
            if any(t['annotation_id'] not in before for t in truth):
                raise ValueError('Crop introduced an annotation')
            added_x.append(x[index] if arm == 'control' else cv2.resize(image, (160, 120), interpolation=cv2.INTER_AREA))
            added_y.append(y[index] if arm == 'control' else target)
            records.append(dict(source=row['source'], source_sha256=row['source_sha256'], index=index,
                source_group=groups[index], fold=int(folds[index]), window=list(window),
                transformed_truth=truth, collisions=collisions))
    if len(records) != 320 or any(r['source_group'] in {g for g, f in zip(groups, base_folds) if f == 0} for r in records):
        raise ValueError('Augmentation count or held-out source isolation changed')
    result_x = np.concatenate([x, np.asarray(added_x, np.float32)])
    result_y = np.concatenate([y, np.asarray(added_y, np.float32)])
    result_folds = np.concatenate([folds, [r['fold'] for r in records]])
    sources = {**dependencies(), str(PROTOCOL): sha256(PROTOCOL), str(BASE/'manifest.json'): BASE_SHA,
        'scripts/edge_crop_localizer_data.py': sha256(__file__),
        'scripts/balloon_dense_complete.py': sha256('scripts/balloon_dense_complete.py'),
        'scripts/prepare_balloon_dense_proposals.py': sha256('scripts/prepare_balloon_dense_proposals.py')}
    return dict(images=result_x, targets=result_y, folds=result_folds, groups=groups, extra=extra,
        records=records, source_sha256=sources, clipped_visible_boxes=clipped,
        hashes=dict(images=tensor_hash(result_x), targets=tensor_hash(result_y), folds=tensor_hash(result_folds)),
        original_images=x, original_targets=y, original_folds=folds)
