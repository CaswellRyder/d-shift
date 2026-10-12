"""Complete reviewed FIT sources only; frozen localizer and exact proposal labels."""
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256
from scripts.balanced_localizer_data import tensors, dependencies as base_dependencies
from scripts.train_balloon_grid_convergence import source_data
from scripts.cached_balloon_decoder import decode
from scripts.proposal_quality_features import features, quality_target

PROTOCOL = Path('configs/proposal-quality-localizer-20261011.json')


def checked():
    p = read_json(PROTOCOL)
    if p['training_approved'] is not True or any(p[k] is not False for k in
        ('original_holdout_evaluated', 'development_evaluated', 'test_evaluated', 'deployment_approved', 'qualification_achieved')):
        raise ValueError('TRAIN-only scope required')
    if sha256(p['parent_audit']) != p['parent_audit_sha256']:
        raise ValueError('Audited parent changed')
    audit = read_json(p['parent_audit'])
    for path, digest in audit['inputs_sha256'].items():
        if sha256(path) != digest:
            raise ValueError('Parent source changed')
    data = tensors('balanced')
    doc, _, _, folds, _ = source_data('coarse')
    # checked_data recursively verifies this original complete-scene review.
    base_protocol = read_json('configs/balloon-dense-public-trial-20261011.json')
    base_doc = read_json(Path(base_protocol['base_data'])/'manifest.json')
    if base_doc['complete_background']['review'] != p['complete_review']:
        raise ValueError('Wrong complete review')
    review = read_json(p['complete_review'])
    if sha256(p['complete_review']) != base_doc['complete_background']['review_sha256']:
        raise ValueError('Complete annotation review changed')
    complete_sources = {base_doc['frames'][i]['source'] for i in review['complete_indices']}
    records = []
    for index in data['original_fit_indices']:
        if index < len(doc['frames']):
            row = doc['frames'][index]
            if folds[index] == 0:
                raise ValueError('Original holdout entered fitting')
            records.append(dict(source=row['source'], truth=row['truth'], complete=row['source'] in complete_sources))
        else:
            row = data['negative_frames'][index-len(doc['frames'])]
            records.append(dict(source=row['source'], truth=[], complete=True))
    records.extend(dict(source=r['source'], truth=r['truth'], complete=True) for r in data['additional_frames'])
    if len(records) != 139:
        raise ValueError('Source alignment changed')
    records.extend(records[i] for i in data['appended_indices'])
    if len(records) != len(data['x']) or any('IMG_' in r['source'] for r in records):
        raise ValueError('Unapproved family or source alignment')
    return p, audit, data, records


def load_parent(audit, seed):
    import keras
    record = audit['training'][f'presence{seed}']
    root = Path(record['run'])
    for name, digest in record['artifacts_sha256'].items():
        if sha256(root/name) != digest:
            raise ValueError('Parent artifact changed')
    parent = keras.models.load_model(root/'proposal.keras', compile=False)
    parent.trainable = False
    extractor = keras.Model(parent.input, [parent.output, parent.get_layer('separable_conv2d_2').output])
    if parent.count_params() != 2342 or parent.trainable_weights:
        raise ValueError('Require unchanged frozen backbone')
    return parent, extractor, root


def build(seed):
    p, audit, data, records = checked()
    parent, extractor, root = load_parent(audit, seed)
    values, maps = [v.numpy() for v in extractor(data['x'], training=False)]
    np.testing.assert_allclose(values, np.load(root/'predictions.npy', allow_pickle=False), atol=1e-5, rtol=0)
    x, y, identities = [], [], []
    skipped = []
    for index, record in enumerate(records):
        if not record['complete']:
            skipped.append(index)
            continue
        for candidate, proposal in enumerate(decode(values[index], True)):
            if proposal['proposal_score'] < .5:
                continue
            x.append(features(maps[index], proposal))
            y.append(quality_target(proposal, record['truth'], True))
            identities.append(dict(frame_index=index, source=record['source'], candidate=candidate,
                                   proposal=proposal, quality=y[-1]))
    x, y = np.asarray(x, np.float32), np.asarray(y, np.float32)[:, None]
    if not len(x) or x.shape[1] != 31 or not np.isfinite(x).all() or not np.any(y >= .5) or not np.any(y < .5):
        raise ValueError('Require finite quality features with both good and bad boxes')
    mean, scale = x.mean(0), np.maximum(x.std(0), np.float32(.001))
    weights = np.where(y[:, 0] >= .5, p['positive_weight'], 1.).astype(np.float32)
    return dict(x=x, y=y, normalized=(x-mean)/scale, mean=mean, scale=scale, weights=weights,
        identities=identities, records=records, excluded_incomplete_indices=skipped,
        approved_complete_sources=sorted({r['source'] for r in records if r['complete']}),
        parent_sha256=sha256(root/'proposal.keras'), settings=p)


def dependencies():
    p = read_json(PROTOCOL)
    paths = [PROTOCOL, __file__, 'scripts/proposal_quality_features.py', 'scripts/train_proposal_quality_localizer.py',
        'scripts/cached_balloon_decoder.py', 'scripts/research_blue_peak_blend.py', 'src/dtr/tracking.py', p['complete_review'], p['parent_audit']]
    return {**base_dependencies(), **read_json(p['parent_audit'])['inputs_sha256'],
            **{str(path): sha256(path) for path in paths}}


def run_path(seed):
    if seed not in (47, 48):
        raise ValueError('Require fixed seed')
    return Path(f'runs/proposal-quality-localizer-seed{seed}-20261011')
