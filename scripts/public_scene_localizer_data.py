"""Match inputs and schedules while varying only reviewed public scene supervision."""
import hashlib
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256
from scripts.blue_scene_localizer_data import data as previous_data, dependencies as previous_dependencies
from scripts.audit_scale_balanced_search import checked_panels
from scripts.balloon_dense_complete import encode_complete
from scripts.balloon_dense_runtime import prepare_input
from scripts.audit_blue_balloon_window1 import checked_extra as checked_window1
from scripts.audit_blue_balloon_window2 import checked_extra as checked_window2

PROTOCOL = Path('configs/public-scene-localizer-20261011.json')


def supervise(truth, arm):
    if arm not in ('control', 'scenes'):
        raise ValueError('Undeclared training arm')
    target, collisions = encode_complete(truth, [])
    if collisions:
        raise ValueError('Do not silently drop overlapping center supervision')
    if arm == 'control':
        target[..., 2:4] = 0
        target[..., 8] = 0
    return target


def tensors(arm):
    p = read_json(PROTOCOL)
    if arm not in p['arms'] or p['training_approved'] is not True or sha256(p['parent']) != p['parent_sha256']:
        raise ValueError('Unapproved arm or changed parent')
    if any(p[k] is not False for k in ('development_evaluated', 'test_evaluated', 'deployment_approved', 'qualification_achieved')):
        raise ValueError('Only TRAIN localization fitting authorized')
    if any(sha256(path) != digest for path, digest in p['reviews_sha256'].items()):
        raise ValueError('Reviewed labels changed')
    x, y, folds, negatives, groups = previous_data('background')
    keep = folds != 0
    if len(x) != 137 or int(keep.sum()) != 121 or int((~keep).sum()) != 16:
        raise ValueError('Original split changed')
    fit_indices = np.flatnonzero(keep).tolist()
    x, y = x[keep], y[keep]
    panels = checked_panels()
    for name, approved in (('window1', checked_window1()), ('window2', checked_window2())):
        if {r['frame_id'] for r in approved} != {f['id'] for f, _ in panels[name]}:
            raise ValueError('Full-scene sources differ from license/duplicate-checked admissions')
    additional, pixels, targets = [], [], []
    for panel in ('window1', 'window2'):
        for f, rgb in panels[panel]:
            if not f['source'].startswith('openimages/train/') or 'IMG_' in f['source']:
                raise ValueError('Public TRAIN sources only')
            additional.append(dict(panel=panel, **{k: f[k] for k in ('id', 'source', 'frame_sha256', 'truth')}))
            pixels.append(prepare_input(rgb)[0])
            targets.append(supervise(f['truth'], arm))
    if len(additional) != 18 or len({r['source'] for r in additional}) != 18 or sum(len(r['truth']) for r in additional) != 40:
        raise ValueError('Require18distinct sources and40reviewed targets')
    if {r['source'] for r in additional} & {r['source'] for r in negatives}:
        raise ValueError('Positive scene overlaps complete negative source')
    return dict(x=np.concatenate([x, pixels]), y=np.concatenate([y, targets]), settings=p,
        original_fit_indices=fit_indices, original_groups=groups, negative_frames=negatives,
        additional_frames=additional, original_folds=folds.tolist(), panels=panels)


def tensor_hash(a):
    return hashlib.sha256(a.tobytes()).hexdigest()


def matched(a, b):
    if a['x'].shape != (139, 120, 160, 3) or a['y'].shape != (139, 15, 20, 9):
        raise ValueError('Unexpected training tensor dimensions')
    if not np.array_equal(a['x'], b['x']) or not np.array_equal(a['y'][:121], b['y'][:121]):
        raise ValueError('Inputs or original supervision differ across arms')
    if not np.array_equal(a['y'][121:, ..., [0, 1, 4, 5, 6, 7]], b['y'][121:, ..., [0, 1, 4, 5, 6, 7]]):
        raise ValueError('Only loss masks may differ across arms')
    if np.any(a['y'][121:, ..., [2, 3, 8]]) or not np.any(b['y'][121:, ..., 8]):
        raise ValueError('Control must ignore new labels; scenes must supervise reviewed targets')
    for key in ('original_fit_indices', 'original_groups', 'negative_frames', 'additional_frames', 'original_folds', 'settings'):
        if a[key] != b[key]:
            raise ValueError('Arm identity or schedule mismatch')


def dependencies():
    paths = [PROTOCOL, __file__, 'scripts/train_public_scene_localizer.py',
        'scripts/audit_scale_balanced_search.py', 'scripts/window2_scene_labels.py',
        'scripts/balloon_dense_complete.py', 'scripts/prepare_balloon_dense_proposals.py',
        'scripts/blue_scene_localizer_data.py', 'configs/blue-scene-localizer-20261011.json',
        'scripts/audit_blue_balloon_window1.py', 'scripts/audit_blue_balloon_window2.py',
        'scripts/reviewed_balloon_window.py',
        *read_json(PROTOCOL)['reviews_sha256']]
    return {**previous_dependencies(), **{str(p): sha256(p) for p in paths}}


def run_path(arm, seed):
    if arm not in ('control', 'scenes') or seed not in (47, 48):
        raise ValueError('Undeclared run')
    return Path(f'runs/public-scene-localizer-{arm}-seed{seed}-20261011')
