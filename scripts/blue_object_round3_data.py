"""Exact reviewed blue-object negatives with source, license and pixel guards."""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from dtr.data import read_json, sha256
from scripts.acquire_blue_objects_round3 import OUTPUT as ROOT
from scripts.prepare_blue_object_round3_review import OUTPUT as QUEUE
from scripts.prepare_openimages_balloon_review import checked_path
from scripts.openimages_balloon_training import review_labels
from scripts.prepare_grid_clutter_review import guarded_background

REVIEW = Path('configs/blue-object-round3-review-20261011.json')
REVIEW_SHA = '9728b63419bb84a54f301b71cb3d8b8a07b1d13b30dd7aa04e99a2f977cdb00d'


def checked_extra():
    if sha256(REVIEW) != REVIEW_SHA:
        raise ValueError('Sealed visual review changed')
    review,queue = read_json(REVIEW),read_json(QUEUE/'review.json')
    acquisition,licenses = read_json(ROOT/'acquisition.json'),read_json(ROOT/'license-verification.json')
    if (review['queue']!=str(QUEUE) or review['queue_sha256']!=sha256(QUEUE/'review.json')
            or review['reviewed_sheets']!=queue['sheets']
            or queue['script_sha256']!=sha256('scripts/prepare_blue_object_round3_review.py')
            or queue['acquisition_sha256']!=sha256(ROOT/'acquisition.json')
            or queue['license_sha256']!=sha256(ROOT/'license-verification.json')
            or licenses['acquisition_sha256']!=sha256(ROOT/'acquisition.json')
            or acquisition['script_sha256']!=sha256('scripts/acquire_blue_objects_round3.py')):
        raise ValueError('Review/source identity changed')
    checks = {**queue['reference_inputs'],**queue['helpers_sha256'],**acquisition['helpers_sha256']}
    for key,name in (('selection_sha256','selection.json'),('attribution_sha256','attribution.json')):
        checks[str(ROOT/name)] = acquisition[key]
    if any(sha256(p)!=h for p,h in checks.items()):
        raise ValueError('Source helpers, references or metadata changed')
    for name,h in queue['sheets'].items():
        checked_path(QUEUE,name,h)
    for identity in (queue['classifier'],queue['search']):
        path=Path(identity['path'])
        if sha256(path)!=identity['sha256'] or sha256(path.with_suffix('.json'))!=identity['metadata_sha256']:
            raise ValueError('Mining model changed')
    labels=review_labels(review,len(queue['samples']))
    if len(labels)!=13 or set(labels.values())!={'background'}:
        raise ValueError('Require13 reviewed negative crops')
    verified={r['image_id'] for r in licenses['results'] if r['verified'] is True}
    sources={r['image_id']:r for r in acquisition['images']}
    guards={g['frame_id']:[dict(box=g['box'])] for g in review['manual_balloon_guards']}
    rows=[]
    for i,label in sorted(labels.items()):
        r=queue['samples'][i]
        f=queue['frames'][r['frame_id']]
        source=sources[f['image_id']]
        if (r['id']!=i or r['split']!='train' or r['source_image']!='openimages/train/'+f['image_id']
                or f['image_id'] not in verified or f['attribution']!=source['attribution']
                or f['attribution']['Subset']!='train' or f['original_sha256']!=source['sha256']
                or f['original_path']!=source['path'] or f['reference_hamming']<=8
                or (f['selected_hamming'] is not None and f['selected_hamming']<=8)
                or r['mining_prediction']['label'] not in ('red_balloon', 'blue_balloon')
                or not guarded_background(r['proposal']['crop_box'],guards.get(f['id'],[]))):
            raise ValueError('Unsafe or changed source/crop')
        with Image.open(checked_path(ROOT,f['original_path'],f['original_sha256'])) as im:
            original=np.asarray(im.convert('RGB'))
        with Image.open(checked_path(QUEUE,f['path'],f['sha256'])) as im:
            scan=np.asarray(im.convert('RGB'))
        if not np.array_equal(scan,cv2.resize(original,(320,240),interpolation=cv2.INTER_AREA)):
            raise ValueError('Frame differs from source')
        path=checked_path(QUEUE,r['path'],r['sha256'])
        a,b,c,d=r['proposal']['crop_box']
        with Image.open(path) as im:
            if not np.array_equal(np.asarray(im.convert('RGB')),scan[b:d,a:c]):
                raise ValueError('Crop differs from reviewed frame')
        rows.append(dict(r,label=label,file=str(path),preserve=False,view='original',attribution=f['attribution']))
    if len({r['sha256'] for r in rows})!=len(rows):
        raise ValueError('Duplicate admitted pixels')
    return rows

