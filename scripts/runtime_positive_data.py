"""Hash-bound current-search positive views, matched to same-color controls."""
from pathlib import Path

import numpy as np
from PIL import Image

from dtr.data import read_json,sha256
from scripts.audit_runtime_positive_coverage import QUEUE,digest
from scripts.train_current_color_clutter import inputs as base_inputs
from scripts.prepare_blue_false_alarm_review import source_frames
from scripts.prepare_openimages_balloon_review import checked_path
from scripts.openimages_balloon_training import review_labels
from scripts.build_red_blue_proposal_positives import matched_target
from scripts.grid_clutter_data import load_crop

REVIEW = Path('configs/runtime-positive-review-20261011.json')
REVIEW_SHA = 'c2618be3585db6321787fa7379211db0ebab0c50083a0c7a5f416c44adef00de'


def checked_extra(known):
    review,queue = read_json(REVIEW),read_json(QUEUE/'review.json')
    if (sha256(REVIEW)!=REVIEW_SHA or review['queue_sha256']!=sha256(QUEUE/'review.json')
            or review['queue']!=str(QUEUE) or review['reviewed_sheets']!=queue['sheets']
            or queue['script_sha256']!=sha256('scripts/audit_runtime_positive_coverage.py')
            or any(sha256(p)!=h for p,h in queue['helpers_sha256'].items())):
        raise ValueError('Frozen review changed')
    for name,h in queue['sheets'].items():
        checked_path(QUEUE,name,h)
    for identity in [queue['search'],*queue['classifiers'].values()]:
        path = Path(identity['path'])
        if sha256(path)!=identity['sha256'] or sha256(path.with_suffix('.json'))!=identity['metadata_sha256']:
            raise ValueError('Mining identity changed')
    labels = review_labels(review,len(queue['samples']))
    if len(labels)!=167 or set(labels.values())!={'red_balloon','blue_balloon'}:
        raise ValueError('Expected167 reviewed positives')
    sources = {r['source']:(r,rgb) for r,rgb in source_frames() if r['origin']=='complete_fit'}
    extra = []
    for i,label in sorted(labels.items()):
        row = queue['samples'][i]
        frame = queue['frames'][row['frame_id']]
        source,rgb = sources[row['source']]
        targets = [dict(t,scan_box=t['box'],annotation_id=j) for j,t in enumerate(source['truth'])]
        target = matched_target(row['proposal'],targets)
        if (row['id']!=i or row['split']!='train' or target!=row['target'] or target['label']!=label
                or any(frame[k]!=v for k,v in source.items())):
            raise ValueError('Invalid positive match/source')
        with Image.open(checked_path(QUEUE,frame['path'],frame['sha256'])) as im:
            if not np.array_equal(np.asarray(im.convert('RGB')),rgb):
                raise ValueError('Frame pixels changed')
        path = checked_path(QUEUE,row['path'],row['sha256'])
        a,b,c,d = row['proposal']['crop_box']
        with Image.open(path) as im:
            if not np.array_equal(np.asarray(im.convert('RGB')),rgb[b:d,a:c]):
                raise ValueError('Crop pixels changed')
        item = dict(row,label=label,file=str(path),source_image=row['source'],view='original',preserve=True)
        tensor,_ = load_crop(item)
        h = digest(tensor)
        if h!=row['tensor_sha256'] or bool(h in known)!=row['in_training']:
            raise ValueError('Training tensor identity changed')
        if h not in known:
            extra.append(item)
    if len(extra)!=139 or sum(r['label']=='red_balloon' for r in extra)!=70:
        raise ValueError('Expected70red69blue new inputs')
    return extra


def inputs(mode):
    if mode not in ('control','admit'):
        raise ValueError('Require paired arm')
    rows,labels,ordinary,tensors = base_inputs('admit')
    extra = checked_extra({digest(x) for x in tensors[0]})
    rng = np.random.default_rng(47)
    chosen = []
    for color in ('red_balloon','blue_balloon'):
        targets = [r for r in extra if r['label']==color]
        pool = [r for r in rows if r['label']==color and r['preserve']]
        control = [pool[int(i)] for i in rng.choice(len(pool),len(targets),replace=False)]
        chosen.extend(targets if mode=='admit' else control)
    added = [dict(r,preserve=True,view='original') for r in chosen]
    x = np.stack([load_crop(r)[0] for r in added])
    tensors[0] = np.concatenate((tensors[0],x))
    tensors.append(np.stack([load_crop(r)[0] for r in extra]))
    labels = np.concatenate((labels,[1 if r['label']=='red_balloon' else 2 for r in added]))
    ordinary = np.concatenate((ordinary,np.ones(len(added),bool)))
    return rows+added,labels,ordinary,tensors,np.array([1 if r['label']=='red_balloon' else 2 for r in extra])
