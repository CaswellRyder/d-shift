"""Matched TRAIN-only context inputs with unchanged original-crop teacher targets."""
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from dtr.data import read_json, sha256
from scripts.clutter_round3_feature_data import tensors as original_tensors, hashes as original_hashes
from scripts.balloon_wider_crop import wider_crop
from scripts.blue_object_round3_data import QUEUE
from scripts.prepare_openimages_balloon_review import checked_path
from scripts.prepare_blue_false_alarm_review import source_frames

PROTOCOL=Path('configs/balloon-context-distillation-20261011.json')


def resized(rgb, box):
    a,b,c,d=box
    return np.asarray(Image.fromarray(rgb[b:d,a:c]).resize((64,64),Image.Resampling.BILINEAR),np.float32)


def tensors(arm):
    p=read_json(PROTOCOL)
    if arm not in p['arms']:
        raise ValueError('Unknown context arm')
    base=original_tensors('admit')
    keys=('parent','parent_sha256','seeds','epochs','batch_size','learning_rate','cosine_alpha',
          'hard_weight','parameters','new_repetitions','rows','optimizer_steps','scene_kl_weight','scene_centered_logit_weight')
    if any(base['p'][k]!=p[k] for k in keys):
        raise ValueError('Context/control recipe mismatch')
    base['p']=p
    base['teacher_x']=base['x'].copy()
    base['teacher_probes']=base['probes'].copy()
    queue=read_json(QUEUE/'review.json')
    if len(queue['samples'])!=13 or [r['id'] for r in queue['samples']]!=list(range(13)):
        raise ValueError('Review ordering changed')
    new,identities=[],[]
    for i,r in enumerate(queue['samples']):
        f=queue['frames'][r['frame_id']]
        with Image.open(checked_path(QUEUE,f['path'],f['sha256'])) as im:
            rgb=np.asarray(im.convert('RGB'))
        old=r['proposal']['crop_box']
        if not np.array_equal(resized(rgb,old),base['probes'][i]):
            raise ValueError('Original probe pixels do not reproduce')
        box=old if arm=='control' else wider_crop(old,rgb.shape)
        new.append(resized(rgb,box))
        identities.append(dict(source=r['source_image'],frame_sha256=f['sha256'],original_crop=old,crop_box=box))
    base['probes']=np.stack(new)
    base['x'][2074:2126]=np.concatenate([base['probes']]*4)
    scenes={s['source']:(s,rgb) for s,rgb in source_frames() if s['origin']=='complete_fit'}
    if len(scenes)!=64 or any('IMG_' in name for name in scenes):
        raise ValueError('TRAIN-only scene boundary changed')
    scene_context=[]
    for i,r in enumerate(base['scene_ids']):
        s,rgb=scenes[r['source']]
        if s['source_file_sha256']!=r['frame_sha256'] or not np.array_equal(resized(rgb,r['crop_box']),base['teacher_x'][2126+i]):
            raise ValueError('Original TRAIN scene crop does not reproduce')
        box=r['crop_box'] if arm=='control' else wider_crop(r['crop_box'],rgb.shape)
        base['x'][2126+i]=resized(rgb,box)
        scene_context.append(dict(source=r['source'],frame_sha256=r['frame_sha256'],
            proposal_index=r['proposal_index'],original_crop=r['crop_box'],crop_box=box))
    base['context_probes']=identities
    base['context_scenes']=scene_context
    for path in (PROTOCOL,__file__,'scripts/balloon_wider_crop.py','scripts/prepare_blue_false_alarm_review.py'):
        base['sources_sha256'][str(path)]=sha256(path)
    if arm=='control' and (not np.array_equal(base['x'],base['teacher_x']) or not np.array_equal(base['probes'],base['teacher_probes'])):
        raise ValueError('Control changed pixels')
    return base


def hashes(data):
    return dict(original_hashes(data),**{k:hashlib.sha256(data[k].tobytes()).hexdigest()
                for k in ('teacher_x','teacher_probes')})
