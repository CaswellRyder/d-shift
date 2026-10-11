"""Matched replay schedules with different exposure to already-reviewed TRAIN scenes."""
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256
from scripts.public_scene_localizer_data import tensors as base_tensors, dependencies as base_dependencies

PROTOCOL = Path('configs/balanced-localizer-20261011.json')


def extra_indices(arm):
    if arm == 'control':
        return np.random.default_rng(47).choice(121, 54, replace=False)
    if arm == 'balanced':
        return np.tile(np.arange(121, 139), 3)
    raise ValueError('Undeclared arm')


def expand(x, y, arm):
    if len(x) != 139 or len(y) != 139:
        raise ValueError('Only the verified139FITrows may enter this trial')
    chosen = extra_indices(arm)
    return np.concatenate([x, x[chosen]]), np.concatenate([y, y[chosen]])


def tensors(arm):
    p = read_json(PROTOCOL)
    if (p['training_approved'] is not True or arm not in p['arms']
            or sha256(p['source_protocol']) != p['source_protocol_sha256']
            or sha256(p['parent']) != p['parent_sha256']
            or any(p[k] is not False for k in ('original_holdout_evaluated', 'development_evaluated',
                'test_evaluated', 'deployment_approved', 'qualification_achieved'))):
        raise ValueError('Changed parent, source protocol or TRAIN scope')
    base = base_tensors('scenes')
    x, y = expand(base['x'], base['y'], arm)
    if x.shape != (193, 120, 160, 3) or y.shape != (193, 15, 20, 9):
        raise ValueError('Training shape changed')
    return dict(base, x=x, y=y, arm=arm, base_settings=base['settings'], settings=p,
                appended_indices=extra_indices(arm).tolist())


def matched(a, b):
    for key in ('original_fit_indices', 'original_groups', 'negative_frames', 'additional_frames',
                'original_folds', 'settings', 'base_settings'):
        if a[key] != b[key]:
            raise ValueError('Matched sources or schedule differ')
    if not np.array_equal(a['x'][:139], b['x'][:139]) or not np.array_equal(a['y'][:139], b['y'][:139]):
        raise ValueError('Original supervision or pixels changed')
    for arm, d in (('control', a), ('balanced', b)):
        expected = extra_indices(arm)
        if d['appended_indices'] != expected.tolist():
            raise ValueError('Replay sample allocation changed')
        for key in ('x', 'y'):
            if not np.array_equal(d[key][139:], d[key][expected]):
                raise ValueError('Replay rows are not exact copies of approved inputs')


def dependencies():
    return {**base_dependencies(), **{str(p): sha256(p) for p in (PROTOCOL, __file__,
        'scripts/train_balanced_localizer.py', 'scripts/balloon_relative_box_loss.py')}}


def run_path(arm, seed):
    if arm not in ('control', 'balanced') or seed not in (47, 48):
        raise ValueError('Undeclared run')
    return Path(f'runs/balanced-localizer-{arm}-seed{seed}-20261011')
