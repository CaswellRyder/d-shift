"""Sparse supervision for a research full-frame localizer, not new scene labels.

Only reviewed balloon interiors and explicit reviewed background regions receive
objectness supervision. Unknown pixels are ignored. Reserved IMG views are never
loaded. Masks describe center likelihood, not balloon segmentation.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, write_json
from scripts.audit_balloon_search_stages import original_frames, indoor_frames, training_source
from scripts.evaluate_balloon_edge_search import BASE, ORIGINAL, INDOOR, TRAIN_QUEUES
from scripts.prepare_balloon_instance_review import checked_rows
from scripts.train_balloon_fragments import MANIFEST, QUEUE

GRID = (15, 20)
NEGATIVE_REVIEW = 'configs/balloon-lab-clutter-review-20261011.json'


def encode(truth, negatives):
    """Grid centers in original 320x240 coordinates; six prediction channels.

    Return heat[2], heat_mask[2], offsets[2], size[2], box_mask[1]. A cell with
    competing target centers receives no box supervision. No invented negatives.
    """
    h, w = GRID
    xx, yy = np.meshgrid((np.arange(w)+.5)*16, (np.arange(h)+.5)*16)
    heat = np.zeros((h,w,2), np.float32)
    mask = np.zeros_like(heat)
    box = np.zeros((h,w,4), np.float32)
    box_mask = np.zeros((h,w,1), np.float32)
    occupied = set()
    collision = set()
    for target in truth:
        a,b,c,d = target['box']
        if target['label'] not in ('red_balloon','blue_balloon') or not 0 <= a < c <= 320 or not 0 <= b < d <= 240:
            raise ValueError('Invalid reviewed target')
        group = 0 if target['label'] == 'red_balloon' else 1
        cx,cy = (a+c)/2, (b+d)/2
        gx,gy = min(w-1,int(cx/16)), min(h-1,int(cy/16))
        inside = (xx>=a)&(xx<c)&(yy>=b)&(yy<d)
        # At least the center cell is supervised even for sub-grid balloons.
        inside[gy,gx] = True
        peak = np.exp(-.5*(((xx-cx)/max((c-a)/6,8))**2+((yy-cy)/max((d-b)/6,8))**2))
        peak[gy,gx] = 1.
        heat[:,:,group] = np.maximum(heat[:,:,group], np.where(inside,peak,0))
        mask[inside] = 1.
        if (gy,gx) in occupied:
            collision.add((gy,gx))
        occupied.add((gy,gx))
        box[gy,gx] = [cx/16-gx,cy/16-gy,(c-a)/320,(d-b)/240]
        box_mask[gy,gx] = 1.
    for region in negatives:
        a,b,c,d = region
        if not 0 <= a < c <= 320 or not 0 <= b < d <= 240:
            raise ValueError('Invalid explicit background region')
        # Full grid cell must be inside reviewed background; centers alone could
        # put a negative on neighboring balloon pixels.
        inside = (xx-8>=a)&(xx+8<=c)&(yy-8>=b)&(yy+8<=d)
        if np.any(mask[inside]):
            raise ValueError('Explicit background overlaps supervised balloon cell')
        mask[inside] = 1.
    for gy,gx in collision:
        box_mask[gy,gx] = 0.
    return np.concatenate((heat,mask,box,box_mask),axis=-1), len(collision)


def prepare(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    base = read_json(MANIFEST)
    allowed = {r['source_image']:r['source_sha256'] for r in base['samples'] if r['split']=='train'}
    forbidden = {r['source_sha256'] for r in base['samples'] if r['split']!='train'}
    panels = {'original':original_frames(BASE,ORIGINAL)}
    panels.update({q:indoor_frames('data/'+q,'configs/'+q+'.json',BASE,INDOOR) for q in TRAIN_QUEUES})
    frames = {}
    for panel,items in panels.items():
        for source,rgb,truth in items:
            training_source(source)
            key = source if source in allowed else 'train/'+source
            if key not in allowed or allowed[key] in forbidden or key in frames:
                raise ValueError('Unadmitted, repeated or cross-split source')
            frames[key] = dict(rgb=rgb,truth=truth,negatives=[],source_sha256=allowed[key],panel=panel)
    negatives = checked_rows(MANIFEST,QUEUE,NEGATIVE_REVIEW)
    queue = read_json(Path(QUEUE)/'review.json')
    for negative in negatives:
        if negative['label'] != 'background':
            raise ValueError('Require explicit background review')
        key = negative['source_image']
        qf = queue['frames'][negative['frame_id']]
        path = Path(QUEUE)/qf['path']
        if key not in allowed or sha256(path) != qf['sha256']:
            raise ValueError('Negative frame identity changed')
        with Image.open(path) as image:
            rgb = np.asarray(image.convert('RGB'))
        if key not in frames:
            frames[key] = dict(rgb=rgb,truth=[],negatives=[],source_sha256=allowed[key],panel='reviewed-background-context')
        if not np.array_equal(rgb,frames[key]['rgb']):
            raise ValueError('Negative and positive frame pixels differ')
        frames[key]['negatives'].append(negative['crop_box'])
    output.mkdir(parents=True)
    records = []
    targets = []
    for index,(source,frame) in enumerate(sorted(frames.items())):
        target,collisions = encode(frame['truth'],frame['negatives'])
        path = output/f'{index:03d}.png'
        Image.fromarray(frame['rgb']).save(path)
        records.append(dict(source=source,source_sha256=frame['source_sha256'],panel=frame['panel'],
            path=path.name,sha256=sha256(path),truth=frame['truth'],negative_regions=frame['negatives'],
            supervised_heat_cells=int((target[:,:,2:4]>0).sum()),box_cells=int(target[:,:,-1].sum()),
            box_collisions=collisions))
        targets.append(target)
    np.save(output/'targets.npy',np.stack(targets),allow_pickle=False)
    doc = dict(frames=records,targets_sha256=sha256(output/'targets.npy'),grid=list(GRID),
        channels=['red_heat','blue_heat','red_mask','blue_mask','offset_x','offset_y','width','height','box_mask'],
        source_manifest_sha256=sha256(MANIFEST),negative_review_sha256=sha256(NEGATIVE_REVIEW),
        training_reviews={q:dict(queue_sha256=sha256('data/'+q+'/review.json'),
            review_sha256=sha256('configs/'+q+'.json')) for q in TRAIN_QUEUES},
        input_archives={p:sha256(p) for p in (ORIGINAL,INDOOR)},script_sha256=sha256(__file__),
        scope='Sparse TRAIN labels only; unknown scene areas ignored, not background. No independent qualification.',
        training_approved=True,test_evaluated=False,deployment_approved=False)
    write_json(output/'manifest.json',doc)
    return dict(frames=len(records),positive_boxes=sum(len(r['truth']) for r in records),
                negative_regions=sum(len(r['negative_regions']) for r in records),
                heat_cells=sum(r['supervised_heat_cells'] for r in records),
                box_cells=sum(r['box_cells'] for r in records),
                manifest_sha256=sha256(output/'manifest.json'))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    print(prepare(p.parse_args().output))
