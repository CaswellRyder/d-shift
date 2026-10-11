"""Bounded public TRAIN blue-object quarantine; upstream class is not a label."""
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
from pathlib import Path
import re

import cv2
import numpy as np

from dtr.data import read_json, sha256, write_json
from scripts.acquire_openimages_balloons import BALLOON, BOX_URL, CLASS_URL, bounded_get, metadata_for
from scripts.acquire_more_openimages_balloons import download

OUTPUT = Path('data/raw/openimages-blue-objects-20261011')
CLASSES = {'/m/01n4qj':'Shirt', '/m/0bt_c3':'Book', '/m/04h8sr':'Dumbbell',
           '/m/01940j':'Backpack', '/m/05gqfk':'Plastic bag'}


def object_groups(raw):
    text = raw[:raw.rfind(b'\n')+1].decode('utf-8')
    groups, balloons, last = {}, set(), None
    for row in csv.DictReader(io.StringIO(text)):
        last = row['ImageID']
        if not re.fullmatch('[0-9a-f]{16}', last):
            raise ValueError('Invalid image ID')
        if row['LabelName'] == BALLOON:
            balloons.add(last)
        if row['LabelName'] not in CLASSES:
            continue
        box = [float(row[k]) for k in ('XMin','YMin','XMax','YMax')]
        if not np.isfinite(box).all() or not (0<=box[0]<box[2]<=1 and 0<=box[1]<box[3]<=1):
            raise ValueError('Invalid object box')
        if row['IsDepiction']=='0' and row['IsGroupOf']=='0':
            groups.setdefault(last, []).append(row)
    # Range tail may omit remaining annotations of its final image.
    for identifier in balloons | {last}:
        groups.pop(identifier, None)
    return groups


def choose(groups, excluded, per_class=32):
    if type(per_class) is not int or not 1<=per_class<=32:
        raise ValueError('Require bounded selection')
    selected = {}
    for label in CLASSES:
        candidates = [i for i, rows in groups.items() if i not in excluded and i not in selected
                      and any(r['LabelName']==label for r in rows)]
        candidates.sort(key=lambda i: hashlib.sha256(('blue-object47/'+i).encode()).hexdigest())
        selected.update({i:label for i in candidates[:per_class]})
    return selected


def blue_fraction(rgb, annotations):
    scan = cv2.resize(rgb, (320,240), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(scan, cv2.COLOR_RGB2HSV)
    mask = (hsv[:,:,0]>=90)&(hsv[:,:,0]<=135)&(hsv[:,:,1]>=60)&(hsv[:,:,2]>=30)
    values = []
    for r in annotations:
        a,b,c,d = [float(r[k])*scale for k,scale in zip(('XMin','YMin','XMax','YMax'),(320,240,320,240))]
        a,b,c,d = int(a),int(b),min(320,int(np.ceil(c))),min(240,int(np.ceil(d)))
        n = int(mask[b:d,a:c].sum())
        values.append(dict(label=r['LabelName'],blue_pixels=n,fraction=n/max(1,(c-a)*(d-b))))
    return values


def acquire():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    prior = sorted(Path('data/raw').glob('openimages*/acquisition.json'))
    excluded = {r['image_id'] for path in prior for r in read_json(path)['images']}
    classes,_ = bounded_get(CLASS_URL,1<<20)
    catalog = dict(csv.reader(io.StringIO(classes.decode())))
    if any(catalog.get(k)!=v for k,v in CLASSES.items()):
        raise ValueError('Class catalog changed')
    size = 64<<20
    raw,headers = bounded_get(BOX_URL,size,{'Range':f'bytes=0-{size-1}'},206)
    if len(raw)!=size or not headers.get('Content-Range','').startswith(f'bytes 0-{size-1}/'):
        raise ValueError('Incorrect annotation range')
    groups = object_groups(raw)
    selected = choose(groups,excluded)
    OUTPUT.mkdir(parents=True)
    write_json(OUTPUT/'selection.json',dict(annotations_url=BOX_URL,scan_bytes=size,
        scan_sha256=hashlib.sha256(raw).hexdigest(),headers=headers,class_csv_sha256=hashlib.sha256(classes).hexdigest(),
        classes=CLASSES,selected=selected,boxes={i:groups[i] for i in selected},
        excluded_acquisitions={str(p):sha256(p) for p in prior},
        selection='Deterministic hash-order quota32 per category; known annotated balloon images and incomplete range-tail group excluded. TRAIN only.'))
    del raw
    kept,screened = [],[]
    (OUTPUT/'images').mkdir()
    with ThreadPoolExecutor(max_workers=4) as pool:
        for n,item in enumerate(pool.map(download,selected),1):
            identifier = item['image_id']
            if 'error' in item:
                screened.append(item)
                continue
            stats = blue_fraction(item['rgb'],groups[identifier])
            retain = any(s['blue_pixels']>=96 and s['fraction']>=.08 for s in stats)
            if not retain:
                screened.append(dict(image_id=identifier,reason='No sufficient blue region inside source object boxes',stats=stats))
            else:
                path = OUTPUT/'images'/(identifier+'.jpg')
                path.write_bytes(item['raw'])
                kept.append(dict(image_id=identifier,path=f'images/{identifier}.jpg',sha256=sha256(path),
                    url=item['url'],headers=item['headers'],size=[item['rgb'].shape[1],item['rgb'].shape[0]],
                    category=CLASSES[selected[identifier]],annotations=groups[identifier],blue_screen=stats))
            print(f'Object image {n}/{len(selected)}; retained {len(kept)} for attribution/review',flush=True)
    if not kept:
        raise ValueError('No eligible blue objects')
    metadata,receipt = metadata_for([r['image_id'] for r in kept])
    write_json(OUTPUT/'attribution.json',metadata)
    write_json(OUTPUT/'metadata-source.json',receipt)
    for r in kept:
        r['attribution'] = metadata[r['image_id']]
    write_json(OUTPUT/'acquisition.json',dict(images=kept,screened_out=screened,
        selection_sha256=sha256(OUTPUT/'selection.json'),attribution_sha256=sha256(OUTPUT/'attribution.json'),
        script_sha256=sha256(__file__),helpers_sha256={p:sha256(p) for p in
            ('scripts/acquire_more_openimages_balloons.py','scripts/acquire_openimages_balloons.py')},
        source='Open Images TRAIN / CVDF resized originals',training_approved=False,review_required=True,
        per_image_license_verified=False,duplicate_screened=False,test_evaluated=False,deployment_approved=False,
        scope='Blue pixels and object category select review candidates only. No automatic background labels.'))
    print(dict(acquired=len(kept),screened_out=len(screened),receipt_sha256=sha256(OUTPUT/'acquisition.json')))


if __name__=='__main__':
    acquire()
