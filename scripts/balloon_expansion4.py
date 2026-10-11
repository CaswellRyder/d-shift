"""Fresh reviewed TRAIN errors only; frozen-parent confidence mining, no dev tuning."""
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256
from dtr.runtime import Predictor
from scripts.balloon_expansion3 import partition, BOOTSTRAP, PRIOR as PREVIOUS
from scripts.build_indoor_balloon_training import admitted
from scripts.audit_public_balloon_export import source_group
from scripts.prepare_indoor_balloon_review import ARCHIVE_SHA, family
from scripts.train_balloon_fragments import checked_data, MANIFEST, QUEUE as FRAGMENT_QUEUE

QUEUE = 'data/engdes2-indoor-expansion4-review-20261011'
REVIEW = 'configs/engdes2-indoor-expansion4-review-20261011.json'
PRIOR = PREVIOUS + ('engdes2-indoor-expansion3-review-20261011',)
PARENT = Path('runs/balloon-fragment-control-seed47-20261011/color.fp32.tflite')
CLASSES = ('background','red_balloon','blue_balloon')


def checked_extra():
    root = Path(QUEUE).resolve()
    queue, review = read_json(root/'review.json'), read_json(REVIEW)
    partition(review, len(queue['samples']))
    if (queue['archive_sha256'] != ARCHIVE_SHA
            or sha256('data/raw/engdes2-red-blue-v1-20261008/dataset.zip') != ARCHIVE_SHA
            or queue['excluded_queue_sha256s'] != [sha256(f'data/{q}/review.json') for q in PRIOR]):
        raise ValueError('Expansion source or exclusions changed')
    forbidden = {source_group(r['source']) for q in PRIOR
                 for r in read_json(f'data/{q}/review.json')['frames']}
    base = read_json(MANIFEST)
    hold = [r for r in base['samples'] if r['split'] != 'train']
    rows = admitted(QUEUE, REVIEW, BOOTSTRAP)
    for row in rows:
        frame = queue['frames'][row['frame_id']]
        path = (root/frame['path']).resolve()
        if (source_group(frame['source']) in forbidden or family(frame['source']) is None
                or frame['prior_training_hamming'] is None or frame['prior_training_hamming'] <= 8
                or not path.is_relative_to(root) or sha256(path) != frame['sha256']):
            raise ValueError('Repeated, reserved or changed expansion frame')
        if any(row[k] == h[k] for h in hold for k in ('sha256','source_image','source_sha256')):
            raise ValueError('Expansion crosses heldout identity')
    known = {r['sha256']:r['label'] for r in base['samples'] if r['split']=='train'}
    if any(r['sha256'] in known and known[r['sha256']] != r['label'] for r in rows):
        raise ValueError('Expansion conflicts with existing label')
    return rows


def hard_examples(rows, predictor):
    """Fixed rule: probability for reviewed class below 0.8, not dev-selected."""
    selected, audit = [], []
    for row in rows:
        prediction = predictor.predict(Path(QUEUE)/row['path'])
        scores = np.asarray(prediction['scores'])
        if scores.shape != (3,) or not np.isfinite(scores).all():
            raise ValueError('Invalid parent scores')
        confidence = float(scores[CLASSES.index(row['label'])])
        hard = confidence < .8
        audit.append(dict(id=row['id'],label=row['label'],reviewed_class_probability=confidence,
                          hard=hard,prediction=prediction))
        if hard:
            selected.append(row)
    return selected, audit


def data(mode):
    if mode not in ('admit','control'):
        raise ValueError('Require matched mode')
    ordinary, fragments = checked_data()
    predictor = Predictor(PARENT, allow_unvalidated=True)
    if tuple(predictor.metadata['classes']) != CLASSES:
        raise ValueError('Parent classes changed')
    extra, _ = hard_examples(checked_extra(), predictor)
    if not extra:
        raise ValueError('No reviewed hard training examples; do not train empty trial')
    rows = [dict(r,file=str(Path(MANIFEST).parent/r['path']),preserve=True) for r in ordinary]
    rows += [dict(r,label='blue_balloon',file=str(Path(FRAGMENT_QUEUE)/r['path']),preserve=True)
             for r in fragments]*4
    pools = {label:[r for r in rows if r['label']==label] for label in CLASSES}
    rng = np.random.default_rng(47)
    additions = []
    for _ in range(4):
        for row in extra:
            choices = pools[row['label']]
            selected = (dict(row,file=str(Path(QUEUE)/row['path'])) if mode=='admit'
                        else choices[int(rng.integers(len(choices)))])
            additions.append(dict(selected,preserve=False))
    return rows+additions, extra
