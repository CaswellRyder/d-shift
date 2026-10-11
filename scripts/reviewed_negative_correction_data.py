"""Matched TRAIN data with explicit, noncontradictory reviewed-scene corrections."""
from pathlib import Path

import numpy as np
from PIL import Image

from dtr.data import read_json, sha256
from scripts.window2_retraining_data import tensors as old_tensors
from scripts.admit_direct_hybrid_negatives import checked_extra, REVIEW, OUTPUT as ADMISSION
from scripts.prepare_direct_hybrid_negatives import ROOT
from scripts.public_scene_localizer_data import tensor_hash

PROTOCOL = Path('configs/reviewed-negative-correction-20261011.json')


def correct_scene_targets(y, scene, preserve, weights, positions, hard_weight=4):
    y, scene, preserve, weights = (v.copy() for v in (y, scene, preserve, weights))
    positions = np.asarray(positions, dtype=int)
    if (not len(positions) or len(set(positions)) != len(positions) or
            np.any(positions < 0) or np.any(positions >= len(y)) or
            not np.all(scene[positions]) or np.any(preserve[positions]) or np.any(y[positions] != -1)):
        raise ValueError('Only distinct verified unlabeled scene rows may be corrected')
    y[positions], scene[positions], weights[positions] = 0, False, hard_weight
    return y, scene, preserve, weights


def run_path(arm, seed):
    if arm not in ('control', 'corrected') or seed not in (47, 48):
        raise ValueError('Undeclared run')
    return Path(f'runs/reviewed-negative-correction-{arm}-seed{seed}-20261011')


def tensors(arm, seed):
    p = read_json(PROTOCOL)
    run_path(arm, seed)
    if (p['training_approved'] is not True or sha256(ADMISSION) != p['admission_sha256']
            or str(ADMISSION) != p['admission'] or sha256(p['parent_audit']) != p['parent_audit_sha256']
            or any(p[k] is not False for k in ('development_evaluated', 'test_evaluated', 'pi_tested',
                                             'deployment_approved', 'qualification_achieved'))):
        raise ValueError('Training scope or admissions changed')
    parent = p['parents'][str(seed)]
    if sha256(parent['path']) != parent['sha256']:
        raise ValueError('Parent checkpoint changed')
    parent_evidence = read_json(p['parent_audit'])['results'][f'admit{seed}']
    for filename, digest in parent_evidence['artifacts_sha256'].items():
        if sha256(Path(parent_evidence['run'])/filename) != digest:
            raise ValueError('Audited parent artifact changed')
    data, rows, admission = old_tensors('admit'), checked_extra(), read_json(ADMISSION)
    if admission['rows'] != rows or sha256('scripts/admit_direct_hybrid_negatives.py') != admission['script_sha256']:
        raise ValueError('Admission evidence changed')
    crops = []
    for row in rows:
        with Image.open(row['path']) as im:
            crops.append(np.asarray(im.convert('RGB').resize((64, 64), Image.Resampling.BILINEAR), np.float32))
    extra = np.stack(crops)
    if len(data['x']) != 2607 or np.flatnonzero(data['scene_mask']).tolist() != list(range(2222, 2607)):
        raise ValueError('Base curriculum or scene order changed')
    positions = []
    for conflict in admission['teacher_retention_conflicts']:
        index = conflict['scene_row']
        identity = data['scene_ids'][index]
        row = next(r for r in rows if r['id'] == conflict['reviewed_id'])
        if (identity['source'] != row['source'] or identity['crop_box'] != row['crop_box']
                or identity['frame_sha256'] != row['frame_sha256']):
            raise ValueError('Reviewed correction does not match exact teacher crop')
        j = rows.index(row)
        np.testing.assert_array_equal(data['x'][2222+index], extra[j])
        positions.append(2222+index)
    if [i-2222 for i in positions] != p['corrected_scene_rows']:
        raise ValueError('Expected exactly two declared corrections')
    # Detect contradictory exact pixel duplicates anywhere else in the curriculum.
    lookup = {tensor_hash(crop): i for i, crop in enumerate(extra)}
    matches = []
    for i, crop in enumerate(data['x']):
        if tensor_hash(crop) in lookup:
            if data['y'][i] > 0 or (data['scene_mask'][i] and i not in positions):
                raise ValueError('Unreviewed conflicting duplicate in old curriculum')
            matches.append(i)
    if arm == 'corrected':
        data['y'], data['scene_mask'], data['preserve'], data['weights'] = correct_scene_targets(
            data['y'], data['scene_mask'], data['preserve'], data['weights'], positions, p['hard_weight'])
    n = len(rows)*p['repetitions']
    for key, values in dict(x=np.tile(extra, (p['repetitions'], 1, 1, 1)),
            y=np.zeros(n, int), scene_mask=np.zeros(n, bool), preserve=np.zeros(n, bool),
            weights=np.full(n, p['hard_weight'], np.float32)).items():
        data[key] = np.concatenate([data[key], values])
    if len(data['x']) != p['rows'] or int(np.ceil(p['rows']/p['batch_size']))*p['epochs'] != p['optimizer_steps']:
        raise ValueError('Matched training budget changed')
    dependencies = [PROTOCOL, REVIEW, ADMISSION, ROOT/'review.json', __file__,
        'scripts/train_reviewed_negative_correction.py', 'scripts/clutter_round3_feature_data.py',
        'scripts/admit_direct_hybrid_negatives.py', 'scripts/prepare_direct_hybrid_negatives.py', p['parent_audit']]
    data['sources_sha256'].update({str(path): sha256(path) for path in dependencies})
    data.update(settings=p, arm=arm, seed=seed, parent=parent, corrections=positions,
        reviewed_negatives=rows, negative_pixels=extra, existing_pixel_matches=matches)
    return data


def matched(a, b):
    for key in ('x', 'preserve', 'probes', 'probe_labels', 'negative_pixels'):
        np.testing.assert_array_equal(a[key], b[key])
    for key in ('settings', 'seed', 'parent', 'corrections', 'reviewed_negatives', 'scene_ids',
                'sources_sha256', 'existing_pixel_matches'):
        if a[key] != b[key]:
            raise ValueError('Matched metadata differs')
    for key in ('y', 'scene_mask', 'weights'):
        if np.flatnonzero(a[key] != b[key]).tolist() != a['corrections']:
            raise ValueError('Only declared scene supervision may differ')
