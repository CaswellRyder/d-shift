"""Reviewed real training-view expansion; matched class-count resampling control."""
from pathlib import Path

import numpy as np

from dtr.data import read_json,sha256
from scripts.build_indoor_balloon_training import admitted
from scripts.audit_public_balloon_export import source_group
from scripts.prepare_indoor_balloon_review import ARCHIVE_SHA,family
from scripts.train_balloon_fragments import checked_data,MANIFEST,QUEUE as FRAGMENT_QUEUE

QUEUE = 'data/engdes2-indoor-expansion3-review-20261011'
REVIEW = 'configs/engdes2-indoor-expansion3-review-20261011.json'
BOOTSTRAP = 'data/balloon-red-blue-bootstrap-20261008/manifest.json'
PRIOR = ('engdes2-indoor-review-20261008','engdes2-indoor-expansion-review-20261008',
         'engdes2-indoor-expansion2-review-20261011')


def partition(review,size):
    ids = review['admit']+review['exclude']
    if (any(type(i) is not int for i in ids) or len(ids)!=len(set(ids))
            or set(ids)!=set(range(size)) or review.get('training_approved') is not True
            or review.get('test_evaluated') is not False or review.get('deployment_approved') is not False):
        raise ValueError('Require complete explicit TRAIN-only review')


def checked_extra():
    root = Path(QUEUE).resolve()
    queue,review = read_json(root/'review.json'),read_json(REVIEW)
    partition(review,len(queue['samples']))
    if (queue['archive_sha256']!=ARCHIVE_SHA
            or sha256('data/raw/engdes2-red-blue-v1-20261008/dataset.zip')!=ARCHIVE_SHA
            or queue['excluded_queue_sha256s']!=[sha256(f'data/{q}/review.json') for q in PRIOR]):
        raise ValueError('Expansion source or exclusions changed')
    forbidden = {source_group(r['source']) for q in PRIOR
                 for r in read_json(f'data/{q}/review.json')['frames']}
    base = read_json(MANIFEST)
    hold = [r for r in base['samples'] if r['split']!='train']
    rows = admitted(QUEUE,REVIEW,BOOTSTRAP)
    for row in rows:
        frame = queue['frames'][row['frame_id']]
        path = (root/frame['path']).resolve()
        if (source_group(frame['source']) in forbidden or family(frame['source']) is None
                or frame['prior_training_hamming'] is None or frame['prior_training_hamming']<=8
                or not path.is_relative_to(root) or sha256(path)!=frame['sha256']):
            raise ValueError('Repeated, reserved or changed expansion frame')
        if any(row[k]==h[k] for h in hold for k in ('sha256','source_image','source_sha256')):
            raise ValueError('Expansion crosses heldout identity')
    known = {r['sha256']:r['label'] for r in base['samples'] if r['split']=='train'}
    if any(r['sha256'] in known and known[r['sha256']]!=r['label'] for r in rows):
        raise ValueError('Expansion conflicts with existing label')
    return rows


def data(mode):
    if mode not in ('admit','control'):
        raise ValueError('Require matched mode')
    ordinary,fragments = checked_data()
    extra = checked_extra()
    rows = [dict(r,file=str(Path(MANIFEST).parent/r['path']),preserve=True) for r in ordinary]
    rows += [dict(r,label='blue_balloon',file=str(Path(FRAGMENT_QUEUE)/r['path']),preserve=True)
             for r in fragments]*4
    pools = {label:[r for r in rows if r['label']==label]
             for label in ('background','red_balloon','blue_balloon')}
    rng = np.random.default_rng(47)
    additions = []
    for _ in range(2):
        for row in extra:
            choices = pools[row['label']]
            selected = (dict(row,file=str(Path(QUEUE)/row['path'])) if mode=='admit'
                        else choices[int(rng.integers(len(choices)))])
            additions.append(dict(selected,preserve=False))
    return rows+additions,extra
