"""Matched positive proposal supervision with unchanged pixels and inference graph."""
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256
from scripts.reviewed_negative_correction_data import tensors as previous_tensors, matched as compare
from scripts.positive_proposal_supervision import proposals, LABELS

PROTOCOL = Path('configs/positive-proposal-correction-20261011.json')


def run_path(arm, seed):
    if arm not in ('control', 'corrected') or seed not in (47, 48):
        raise ValueError('Undeclared run')
    return Path(f'runs/positive-proposal-correction-{arm}-seed{seed}-20261011')


def tensors(arm, seed):
    run_path(arm, seed)
    p = read_json(PROTOCOL)
    if (p['training_approved'] is not True or sha256(p['parent_audit']) != p['parent_audit_sha256']
            or any(p[k] is not False for k in ('development_evaluated', 'test_evaluated', 'pi_tested',
                                             'deployment_approved', 'qualification_achieved'))):
        raise ValueError('Parent or TRAIN scope changed')
    parent = p['parents'][str(seed)]
    if sha256(parent['path']) != parent['sha256']:
        raise ValueError('Parent checkpoint changed')
    record = read_json(p['parent_audit'])['training'][f'corrected{seed}']
    for name, digest in record['artifacts_sha256'].items():
        if sha256(Path(record['run'])/name) != digest:
            raise ValueError('Verified parent artifacts changed')
    data = previous_tensors('corrected', seed)
    records, added, excluded = proposals(data)
    if (len(records) != p['positive_proposals'] or len(added) != p['new_positive_inputs']
            or len(excluded) != p['duplicate_public_inputs_excluded']
            or sum(r['panel'] == 'original' for r in records) != p['original_positive_rows']):
        raise ValueError('Annotation-based admission changed')
    n = len(added)
    for key, values in dict(x=added, y=np.full(n, -1, int), scene_mask=np.ones(n, bool),
            preserve=np.zeros(n, bool), weights=np.ones(n, np.float32)).items():
        data[key] = np.concatenate([data[key], values])
    positions = [r['position'] for r in records]
    if len(set(positions)) != len(positions) or not np.all(data['scene_mask'][positions]):
        raise ValueError('Corrections must refer to unique unknown teacher rows')
    if arm == 'corrected':
        for row in records:
            i = row['position']
            if data['preserve'][i] or data['y'][i] != -1:
                raise ValueError('Cannot overwrite existing hard supervision')
            data['y'][i] = LABELS.index(row['label'])
            data['scene_mask'][i] = False
            data['weights'][i] = p['hard_weight']
    if len(data['x']) != p['rows'] or int(np.ceil(p['rows']/p['batch_size']))*p['epochs'] != p['optimizer_steps']:
        raise ValueError('Matched budget changed')
    paths = [PROTOCOL, __file__, p['parent_audit'], 'scripts/positive_proposal_supervision.py',
             'scripts/train_positive_proposal_correction.py', 'scripts/audit_scale_balanced_search.py',
             'research/balloon-red-blue-20261008/direct-hybrid-balloon-20261011.json']
    data['sources_sha256'].update({str(path): sha256(path) for path in paths})
    data.update(settings=p, arm=arm, seed=seed, parent=parent, corrections=positions,
                positive_records=records, duplicate_public_inputs=excluded)
    return data


def matched(a, b):
    compare(a, b)
    if a['positive_records'] != b['positive_records'] or a['duplicate_public_inputs'] != b['duplicate_public_inputs']:
        raise ValueError('Matched annotation admissions differ')
