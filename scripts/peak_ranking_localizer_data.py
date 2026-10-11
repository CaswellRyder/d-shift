"""Identical TRAIN tensors and initialization; only the local peak objective differs."""
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256
from scripts.balanced_localizer_data import tensors as base_tensors, dependencies as base_dependencies

PROTOCOL = Path('configs/peak-ranking-localizer-20261011.json')


def dependencies():
    p = read_json(PROTOCOL)
    paths = [PROTOCOL, __file__, 'scripts/train_peak_ranking_localizer.py',
             'scripts/balloon_peak_ranking_loss.py', p['parent_audit'], p['diagnosis'], p['coverage_diagnosis']]
    return {**base_dependencies(), **{str(path): sha256(path) for path in paths}}


def run_path(arm, seed):
    if arm not in ('control', 'ranked') or seed not in (47, 48):
        raise ValueError('Undeclared run')
    return Path(f'runs/peak-ranking-localizer-{arm}-seed{seed}-20261011')


def tensors(arm, seed):
    run_path(arm, seed)
    p = read_json(PROTOCOL)
    if (p['training_approved'] is not True or any(p[k] is not False for k in
            ('original_holdout_evaluated', 'development_evaluated', 'test_evaluated', 'pi_tested',
             'deployment_approved', 'qualification_achieved'))):
        raise ValueError('Require TRAIN-only scope')
    for key in ('parent_audit', 'diagnosis', 'coverage_diagnosis'):
        if sha256(p[key]) != p[key+'_sha256']:
            raise ValueError('Prospective evidence changed')
    if p['peak_margin_logits'] != .5 or p['peak_weight'] != 1.:
        raise ValueError('Require declared fixed peak objective')
    parent = p['parents'][str(seed)]
    if sha256(parent['path']) != parent['sha256']:
        raise ValueError('Parent checkpoint changed')
    report = read_json(p['parent_audit'])
    for path, digest in report['inputs_sha256'].items():
        if sha256(path) != digest:
            raise ValueError('Parent sources changed')
    record = report['training'][f'balanced{seed}']
    for name, digest in record['artifacts_sha256'].items():
        if sha256(Path(record['run'])/name) != digest:
            raise ValueError('Parent artifacts changed')
    data = base_tensors('balanced')
    if data['x'].shape != (193, 120, 160, 3) or data['y'].shape != (193, 15, 20, 9):
        raise ValueError('Fixed TRAIN tensor shape changed')
    return dict(data, arm=arm, seed=seed, parent=parent, settings=p,
                replay_settings=data['settings'])


def matched(a, b):
    for key in ('x', 'y'):
        np.testing.assert_array_equal(a[key], b[key])
    for key in ('original_fit_indices', 'original_groups', 'negative_frames', 'additional_frames',
                'original_folds', 'settings', 'base_settings', 'replay_settings',
                'appended_indices', 'parent', 'seed'):
        if a[key] != b[key]:
            raise ValueError('Matched inputs, schedule or initialization differ')
