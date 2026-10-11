"""Rebuild raw TRAIN crops before aspect preservation; retain original teacher tensors."""
from pathlib import Path

import numpy as np
from PIL import Image

from dtr.data import read_json,sha256
from scripts.window1_trial_data import tensors as original_tensors
from scripts.window1_runtime_data import checked_batch,choose
from scripts.blue_object_round3_data import checked_extra
from scripts.grid_clutter_data import blue_surface
from scripts.prepare_blue_false_alarm_review import source_frames
from scripts.audit_window1_scene_search import REVIEW as SCENE_REVIEW
from scripts.prepare_openimages_balloon_review import checked_path
from scripts.balloon_aspect_crop import resized

PROTOCOL = Path('configs/balloon-aspect-trial-20261011.json')


def crop(row, aspect):
    if row['split'] != 'train' or 'IMG_' in row['source_image']:
        raise ValueError('TRAIN-only preprocessing boundary')
    with Image.open(row['file']) as im:
        rgb = np.asarray(im.convert('RGB'))
    if row['view'] == 'blue_surface':
        rgb,_ = blue_surface(rgb)
    elif row['view'] not in ('original','horizontal_flip'):
        raise ValueError('Unknown original augmentation')
    values = resized(rgb,aspect)
    return values[:,::-1].copy() if row['view'] == 'horizontal_flip' else values


def batch_scene_pixels(frames, aspect):
    review = read_json(SCENE_REVIEW)
    root = Path(review['queue'])
    if sha256(root/'review.json') != review['queue_sha256']:
        raise ValueError('Changed TRAIN scene review')
    queue = read_json(root/'review.json')
    pixels = []
    for f in frames:
        source = queue['frames'][f['id']]
        if f['source'] != 'openimages/train/'+source['image_id'] or f['frame_sha256'] != source['sha256']:
            raise ValueError('Scene source changed')
        with Image.open(checked_path(root,source['path'],source['sha256'])) as im:
            rgb = np.asarray(im.convert('RGB'))
        for d in f['detections']:
            a,b,c,e = d['crop_box']
            if not 0<=a<c<=rgb.shape[1] or not 0<=b<e<=rgb.shape[0]:
                raise ValueError('Invalid scene crop')
            pixels.append(resized(rgb[b:e,a:c],aspect))
    return np.stack(pixels)


def tensors(arm):
    if arm not in ('control','aspect'):
        raise ValueError('Unknown arm')
    data = original_tensors('hard')
    data['teacher_x'] = data['x'].copy()
    data['teacher_probes'] = data['probes'].copy()
    data['teacher_batch_pixels'] = data['batch_pixels'].copy()
    if arm == 'control':
        return data
    p = read_json(PROTOCOL)
    extra = checked_extra()
    easy,hard,_,sources = checked_batch()
    rows = data['rows']+extra*4+choose(easy,hard,'hard')*4
    if len(rows) != 2162:
        raise ValueError('Supervised inventory changed')
    for i,row in enumerate(rows):
        np.testing.assert_array_equal(crop(row,False),data['teacher_x'][i])
        data['x'][i] = crop(row,True)
    data['probes'] = np.stack([crop(r,True) for r in extra])
    data['batch_pixels'] = np.stack([crop(r,True) for r in data['batch_rows']])
    frames = {s['source']:(s,rgb) for s,rgb in source_frames() if s['origin']=='complete_fit'}
    if len(frames) != 64 or any('IMG_' in s for s in frames):
        raise ValueError('Only64TRAIN scenes allowed')
    for i,r in enumerate(data['scene_ids']):
        s,rgb = frames[r['source']]
        if s['source_file_sha256'] != r['frame_sha256']:
            raise ValueError('Scene identity changed')
        a,b,c,d = r['crop_box']
        np.testing.assert_array_equal(resized(rgb[b:d,a:c],False),data['teacher_x'][2162+i])
        data['x'][2162+i] = resized(rgb[b:d,a:c],True)
    data['p'] = dict(data['p'],arms=p['arms'],recipe=p['recipe'],gate=p['gate'])
    data['protocol'] = p
    data['sources_sha256'].update(sources)
    paths = (PROTOCOL,__file__,'scripts/balloon_aspect_crop.py','scripts/prepare_blue_false_alarm_review.py')
    data['sources_sha256'].update({str(f):sha256(f) for f in paths})
    return data


def matched_inputs(control, aspect):
    for k in ('y','weights','preserve','scene_mask','probe_labels','batch_labels'):
        np.testing.assert_array_equal(control[k],aspect[k])
    for k in ('x','probes','batch_pixels'):
        np.testing.assert_array_equal(control[k],aspect['teacher_'+k])
    if control['sample_identity'] != aspect['sample_identity'] or control['scene_ids'] != aspect['scene_ids']:
        raise ValueError('Input identity changed')
    for k in control['p']:
        if k not in ('arms','recipe','gate') and control['p'][k] != aspect['p'][k]:
            raise ValueError('Schedule changed')
    return np.flatnonzero(np.any(control['x']!=aspect['x'],axis=(1,2,3))).tolist()
