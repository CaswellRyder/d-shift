"""Hash-bound complete-scene TRAIN review and conservative background masks."""
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256
from scripts.prepare_balloon_dense_proposals import encode


def encode_complete(truth, negatives, guard=16):
    if guard != 16:
        raise ValueError('Require fixed 16px uncertainty guard')
    target, collisions = encode(truth, negatives)
    xx, yy = np.meshgrid((np.arange(20)+.5)*16, (np.arange(15)+.5)*16)
    background = np.ones((15,20), bool)
    for row in truth:
        a,b,c,d = row['box']
        # A whole cell must be outside every expanded balloon rectangle.
        intersects = (xx+8>a-guard)&(xx-8<c+guard)&(yy+8>b-guard)&(yy-8<d+guard)
        background &= ~intersects
    target[:,:,2:4][background] = 1.
    return target, collisions


def reviewed_indices(base, folds, groups, queue_path, review_path, base_sha):
    queue_path, review_path = Path(queue_path), Path(review_path)
    queue, review = read_json(queue_path), read_json(review_path)
    if (Path(review['queue']).resolve()!=queue_path.resolve()
            or review['queue_sha256']!=sha256(queue_path)
            or review['source_manifest_sha256']!=base_sha
            or queue['source_manifest_sha256']!=base_sha
            or review.get('guard_pixels')!=16
            or review.get('training_approved') is not True
            or review.get('test_evaluated') is not False
            or review.get('deployment_approved') is not False):
        raise ValueError('Complete review identity or authorization changed')
    for name, digest in queue['sheets'].items():
        path = (queue_path.parent/name).resolve()
        if not path.is_relative_to(queue_path.parent.resolve()) or sha256(path)!=digest:
            raise ValueError('Review sheet changed')
    eligible = {i for i,(r,f) in enumerate(zip(base['frames'],folds,strict=True))
                if f!=0 and r['panel'].startswith('engdes2-')}
    rows = queue['frames']
    if len(rows)!=len(eligible) or {r['index'] for r in rows}!=eligible:
        raise ValueError('Queue includes heldout or unreviewed frames')
    for row in rows:
        i = row['index']
        expected = dict(base['frames'][i],index=i,fold=int(folds[i]),source_group=groups[i])
        if row != expected:
            raise ValueError('Reviewed frame or annotations changed')
    approved = review['complete_indices']
    excluded = {int(i) for i in review['excluded']}
    if (len(set(approved))!=len(approved) or set(approved)&excluded
            or set(approved)|excluded!=eligible
            or any(not reason for reason in review['excluded'].values())):
        raise ValueError('Require exactly one explicit decision for every eligible frame')
    return set(approved)


def validate_derivative(doc):
    """Recheck immutable sparse source, reviews, folds, and unchanged annotations."""
    from scripts.train_balloon_dense_proposals import checked_data
    spec = doc['complete_background']
    root = Path(spec['source_root'])
    if sha256(root/'manifest.json')!=spec['source_manifest_sha256']:
        raise ValueError('Sparse source changed')
    if 'complete_background' in read_json(root/'manifest.json'):
        raise ValueError('Nested background derivatives not allowed')
    base,_,_,folds,groups = checked_data(root)
    if (sha256(spec['review'])!=spec['review_sha256']
            or sha256(spec['queue'])!=spec['queue_sha256'] or spec['guard_pixels']!=16):
        raise ValueError('Derivative review receipt changed')
    approved = reviewed_indices(base,folds,groups,spec['queue'],spec['review'],
                                spec['source_manifest_sha256'])
    if len(doc['frames'])!=len(base['frames']):
        raise ValueError('Derivative frame inventory changed')
    for row, original in zip(doc['frames'],base['frames'],strict=True):
        copied = dict(row,supervised_heat_cells=original['supervised_heat_cells'])
        if copied!=original:
            raise ValueError('Derivative changed source pixels, geometry or order')
    return approved
