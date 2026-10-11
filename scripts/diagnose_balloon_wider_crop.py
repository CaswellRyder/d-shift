"""TRAIN-only context diagnostic using original pixels and the fixed parent."""
import copy
from pathlib import Path

import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, write_json
from dtr.runtime import softmax
from scripts.balloon_wider_crop import wider_crop
from scripts.balloon_scene_constraints import RAW as TRAIN, RAW_SHA
from scripts.blue_object_round3_data import checked_extra, QUEUE
from scripts.clutter_round3_feature_data import PROTOCOL as PARENT_PROTOCOL
from scripts.diagnose_clutter_stage_attribution import crops, difference
from scripts.diagnose_clutter_round3_feature import replay, check_replayed_parent
from scripts.prepare_blue_false_alarm_review import source_frames
from scripts.prepare_openimages_balloon_review import checked_path

PROTOCOL=Path('configs/balloon-wider-crop-20261011.json')
OUTPUT=Path('research/balloon-red-blue-20261008/balloon-wider-crop-20261011.json')
RAW=Path('runs/balloon-wider-crop-20261011.json')


def diagnose():
    if OUTPUT.exists() or RAW.exists():
        raise FileExistsError('Preserve fixed context diagnostic')
    p=read_json(PARENT_PROTOCOL)
    if sha256(p['parent'])!=p['parent_sha256'] or sha256(TRAIN)!=RAW_SHA:
        raise ValueError('Parent or TRAIN reference changed')
    raw=read_json(TRAIN)
    if set(raw)!={'training'}:
        raise ValueError('No DEV fitting or evaluation here')
    extra=checked_extra()
    queue=read_json(QUEUE/'review.json')
    frames=raw['training']['parent']['frames']
    pixels={s['source']:(s['source_file_sha256'],s['truth'],rgb)
            for s,rgb in source_frames() if s['origin']=='complete_fit'}
    if len(pixels)!=64 or any('IMG_' in name for name in pixels):
        raise ValueError('TRAIN boundary changed')
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    model=keras.models.load_model(p['parent'],compile=False)
    def predict(x):
        return softmax(np.concatenate([model(x[i:i+64],training=False).numpy() for i in range(0,len(x),64)]))
    original=replay(frames,predict(crops(frames,pixels)))
    for a,b in zip(original['frames'],frames,strict=True):
        check_replayed_parent(a,b)
    changed=copy.deepcopy(frames)
    for f in changed:
        for d in f['detections']:
            d['crop_box']=wider_crop(d['crop_box'],(240,320,3))
    wider=replay(changed,predict(crops(changed,pixels)))
    probes=[]
    for row in extra:
        f=queue['frames'][row['frame_id']]
        with Image.open(checked_path(QUEUE,f['path'],f['sha256'])) as im:
            rgb=np.asarray(im.convert('RGB'))
        boxes=[row['proposal']['crop_box'],wider_crop(row['proposal']['crop_box'],rgb.shape)]
        x=np.stack([np.asarray(Image.fromarray(rgb[b:d,a:c]).resize((64,64),Image.Resampling.BILINEAR),np.float32)
                    for a,b,c,d in boxes])
        values=predict(x)
        if not np.allclose(values[0],row['mining_prediction']['scores'],atol=1e-5,rtol=0):
            raise ValueError('Original reviewed prediction changed')
        probes.append(dict(id=row['id'],source=row['source_image'],boxes=boxes,scores=values.tolist(),
            accepted=[bool(v.argmax()>0 and v.max()>=.8) for v in values],
            background_correct=[bool(v.argmax()==0) for v in values]))
    effect=difference(wider,original)
    accepted=[sum(r['accepted'][i] for r in probes) for i in (0,1)]
    gate=not effect['lost_targets'] and all(v['fp']<=0 for v in effect['count_delta'].values()) and accepted[1]<accepted[0]
    write_json(RAW,dict(original=original,wider=wider))
    write_json(OUTPUT,dict(protocol_sha256=sha256(PROTOCOL),script_sha256=sha256(__file__),
        parent_sha256=p['parent_sha256'],train_raw_sha256=RAW_SHA,review_queue_sha256=sha256(QUEUE/'review.json'),
        raw=dict(path=str(RAW),sha256=sha256(RAW)),metrics=dict(original=original['metrics'],wider=wider['metrics']),
        effect=effect,probes=probes,background_false_acceptances=accepted,
        eligible_for_development=gate,
        helpers_sha256={s:sha256(s) for s in ('scripts/balloon_wider_crop.py','scripts/blue_object_round3_data.py',
            'scripts/diagnose_clutter_stage_attribution.py','scripts/diagnose_clutter_round3_feature.py')},
        scope='TRAIN-only hypothesis test; negative targets keep their reviewed identity while contextual pixels expand. Same385proposals and classifier calls. Not independent accuracy or Pi timing.',
        training_performed=False,development_evaluated=False,test_evaluated=False,pi_tested=False,
        deployment_approved=False,qualification_achieved=False))
    print(dict(metrics=dict(original=original['metrics'],wider=wider['metrics']),
        lost=len(effect['lost_targets']),gained=len(effect['gained_targets']),
        background_false_acceptances=accepted,eligible_for_development=gate),flush=True)


if __name__=='__main__':
    diagnose()
