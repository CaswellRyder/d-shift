"""Complete reviewed TRAIN negatives; never infer whole-scene labels from crops."""
from pathlib import Path

import numpy as np
from PIL import Image

from dtr.data import read_json, sha256
from scripts.balloon_dense_runtime import prepare_input
from scripts.blue_object_round2_data import checked_extra
from scripts.prepare_openimages_balloon_review import checked_path
from scripts.train_balloon_grid_convergence import source_data

PROTOCOL = Path('configs/blue-scene-localizer-20261011.json')
PREFIX = 'blue-scene-localizer'


def negative_targets(count, arm):
    if arm not in ('control', 'background') or type(count) is not int or count < 1:
        raise ValueError('Require declared arm and positive frame count')
    targets = np.zeros((count, 15, 20, 9), np.float32)
    if arm == 'background':
        targets[..., 2:4] = 1.
    return targets


def data(arm):
    p = read_json(PROTOCOL)
    root = Path(p['queue'])
    if (p['training_approved'] is not True or p['test_evaluated'] is not False
            or p['deployment_approved'] is not False or sha256(root/'review.json') != p['queue_sha256']
            or sha256(p['parent']) != p['parent_sha256']):
        raise ValueError('Protocol, reviewed queue or parent changed')
    queue = read_json(root/'review.json')
    # Recursive source/license/duplicate/crop guards validate every source used here.
    safe_frames = {r['frame_id'] for r in checked_extra()}
    ids = p['complete_empty_frames']
    if len(ids) != 8 or len(set(ids)) != 8 or not set(ids) <= safe_frames or 1 in ids:
        raise ValueError('Complete-scene approval changed or includes ambiguous frame')
    doc, x, y, folds, groups = source_data('coarse')
    if len(x) != 129 or int((folds == 0).sum()) != 16:
        raise ValueError('Original grouped split changed')
    extra, identities = [], []
    for i in ids:
        row = queue['frames'][i]
        if row['id'] != i or row['attribution']['Subset'] != 'train':
            raise ValueError('Not the approved TRAIN frame')
        path = checked_path(root, row['path'], row['sha256'])
        with Image.open(path) as im:
            extra.append(prepare_input(np.asarray(im.convert('RGB')))[0])
        identities.append(dict(frame_id=i, path=str(path), sha256=row['sha256'],
            source='openimages/train/'+row['image_id'], original_sha256=row['original_sha256']))
    if ({r['source'] for r in identities} & {r['source'] for r in doc['frames']}
            or {r['original_sha256'] for r in identities} & {r['source_sha256'] for r in doc['frames']}):
        raise ValueError('New source overlaps original dataset')
    return (np.concatenate([x, extra]), np.concatenate([y, negative_targets(len(extra), arm)]),
            np.concatenate([folds, np.ones(len(extra), dtype=folds.dtype)]), identities, groups)


def run_path(arm, seed):
    if arm not in ('control', 'background') or seed not in (47, 48):
        raise ValueError('Undeclared arm or seed')
    return Path(f'runs/{PREFIX}-{arm}-seed{seed}-20261011')


def dependencies():
    paths = ('scripts/blue_scene_localizer_data.py', 'scripts/blue_object_round2_data.py',
        'scripts/train_balloon_grid_convergence.py', 'scripts/train_balloon_dense_proposals.py',
        'scripts/balloon_dense_iou_loss.py', 'scripts/balloon_dense_runtime.py',
        'configs/blue-object-round2-review-20261011.json', 'configs/balloon-grid-convergence-20261011.json')
    return {p: sha256(p) for p in paths}
