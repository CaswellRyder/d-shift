"""Add approved complete-FIT MSER examples to the frozen learned-proposal corpus."""
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256
from scripts.proposal_quality_data import build as learned_build, checked, load_parent, dependencies as prior_dependencies
from scripts.proposal_quality_features import quality_target
from scripts.shared_proposal_quality import feature_row, candidate
from scripts.audit_scale_balanced_search import checked_panels
from scripts.balloon_dense_runtime import prepare_input

PROTOCOL = Path('configs/shared-proposal-quality-20261011.json')


def checked_previous():
    p = read_json(PROTOCOL)
    if p['training_approved'] is not True or any(p[k] is not False for k in
        ('original_holdout_evaluated', 'development_evaluated', 'test_evaluated', 'deployment_approved', 'qualification_achieved')):
        raise ValueError('Require TRAIN-only scope')
    for name in ('previous_audit', 'diagnosis'):
        if sha256(p[name]) != p[name+'_sha256']:
            raise ValueError('Prospective evidence changed')
    previous = read_json(p['previous_audit'])
    for path, digest in previous['inputs_sha256'].items():
        if sha256(path) != digest:
            raise ValueError('Previous source changed')
    if sha256(previous['raw']['path']) != previous['raw']['sha256']:
        raise ValueError('Frozen parent replay changed')
    return p, previous, read_json(previous['raw']['path'])


def build(seed):
    p, previous, raw = checked_previous()
    learned = learned_build(seed)
    _, parent_audit, data, records = checked()
    _, extractor, _ = load_parent(parent_audit, seed)
    _, maps = [v.numpy() for v in extractor(data['x'], training=False)]
    panel_pixels = {f['source']: (f, rgb) for rows in checked_panels().values() for f, rgb in rows}
    frozen = {f['source']: f for panel in raw.values() for f in panel[f'parent{seed}_independent_neighbor']['frames']}
    if set(panel_pixels) != set(frozen):
        raise ValueError('Reference panel coverage changed')
    x = [np.concatenate([r, np.asarray([1., 0.], np.float32)]) for r in learned['x']]
    y = learned['y'][:, 0].tolist()
    identities = [dict(r, origin='learned') for r in learned['identities']]
    mser_sources = set()
    for index, record in enumerate(records):
        if not record['complete'] or record['source'] not in frozen:
            continue
        frame = frozen[record['source']]
        actual, rgb = panel_pixels[record['source']]
        if record['truth'] != frame['truth'] or any(frame[k] != actual[k] for k in ('id', 'source', 'truth', 'frame_sha256')):
            raise ValueError('Complete training annotation or frame identity differs')
        np.testing.assert_array_equal(prepare_input(rgb)[0], data['x'][index])
        for j, row in enumerate(frame['detections']):
            if row['proposal_origin'] != 'mser' or not row['raw_accepted']:
                continue
            if row['score'] < .8:
                raise ValueError('Crop acceptance threshold changed')
            proposal = candidate(row)
            q = quality_target(proposal, record['truth'], True)
            x.append(feature_row(maps[index], row))
            y.append(q)
            identities.append(dict(frame_index=index, source=record['source'], candidate=j, origin='mser',
                                   proposal=proposal, quality=q, frame_sha256=frame['frame_sha256']))
            mser_sources.add(record['source'])
    x, y = np.asarray(x, np.float32), np.asarray(y, np.float32)[:, None]
    if not mser_sources or len(x) <= len(learned['x']) or x.shape[1] != 33 or not np.isfinite(x).all():
        raise ValueError('Require complete-FIT MSER additions')
    mean, scale = x.mean(0), np.maximum(x.std(0), np.float32(.001))
    weights = np.where(y[:, 0] >= .5, p['positive_weight'], 1.).astype(np.float32)
    return dict(x=x, y=y, normalized=(x-mean)/scale, mean=mean, scale=scale, weights=weights,
        identities=identities, mser_sources=sorted(mser_sources), learned_rows=len(learned['x']),
        complete_sources=learned['approved_complete_sources'], parent_sha256=learned['parent_sha256'], settings=p)


def dependencies():
    p = read_json(PROTOCOL)
    paths = [PROTOCOL, __file__, 'scripts/shared_proposal_quality.py', 'scripts/train_shared_quality.py',
             p['previous_audit'], p['diagnosis']]
    return {**prior_dependencies(), **read_json(p['previous_audit'])['inputs_sha256'],
            **{str(path): sha256(path) for path in paths}}


def run_path(arm, seed):
    if arm not in ('iou', 'valid') or seed not in (47, 48):
        raise ValueError('Undeclared quality arm or seed')
    return Path(f'runs/shared-proposal-quality-{arm}-seed{seed}-20261011')
