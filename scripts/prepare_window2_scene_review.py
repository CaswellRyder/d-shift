"""Expose all source annotations, including nonblue targets, for future complete-scene review."""
from pathlib import Path

import numpy as np
from PIL import Image

from dtr.data import read_json,sha256,write_json
from scripts.audit_blue_balloon_window2 import checked_extra,REVIEW,OUTPUT as ADMISSION
from scripts.prepare_openimages_balloon_review import checked_path,sheets
from scripts.prepare_blue_source_review import crop_box

OUTPUT = Path('data/window2-complete-scene-review-20261011')


def prepare():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    admitted = checked_extra()
    review = read_json(REVIEW)
    root,queue_root = Path(review['root']),Path(review['queue'])
    queue = read_json(queue_root/'review.json')
    acquisition = read_json(root/'acquisition.json')
    sources = {r['image_id']:r for r in acquisition['images']}
    frames,samples = [],[]
    OUTPUT.mkdir(parents=True)
    (OUTPUT/'frames').mkdir()
    (OUTPUT/'crops').mkdir()
    for fid in sorted({r['frame_id'] for r in admitted}):
        frame = queue['frames'][fid]
        with Image.open(checked_path(queue_root,frame['path'],frame['sha256'])) as im:
            rgb = np.asarray(im.convert('RGB'))
        boxes = []
        for index,r in enumerate(sources[frame['image_id']]['annotations']):
            box = [float(r[k])*scale for k,scale in zip(('XMin','YMin','XMax','YMax'),(320,240,320,240))]
            a,b,c,d = crop_box(box)
            if min(c-a,d-b)<1:
                raise ValueError('Empty source annotation')
            path = f'crops/{len(samples):03d}.png'
            Image.fromarray(rgb[b:d,a:c]).save(OUTPUT/path)
            samples.append(dict(id=len(samples),frame_id=fid,source_annotation_index=index,
                source_image='openimages/train/'+frame['image_id'],source_annotation=r,
                path=path,sha256=sha256(OUTPUT/path),box=box,crop_box=[a,b,c,d],split='train',label='unreviewed'))
            boxes.append(box)
        path = f'frames/{fid:03d}.png'
        Image.fromarray(rgb).save(OUTPUT/path)
        frames.append(dict(frame,path=path,sha256=sha256(OUTPUT/path),boxes=boxes))
    sheets(OUTPUT,'frames',frames,3,(320,270))
    sheets(OUTPUT,'crops',samples,6,(160,160))
    write_json(OUTPUT/'review.json',dict(frames=frames,samples=samples,
        inputs_sha256={str(p):sha256(p) for p in (REVIEW,ADMISSION,queue_root/'review.json',root/'acquisition.json',
            __file__,'scripts/prepare_blue_source_review.py','scripts/prepare_openimages_balloon_review.py')},
        sheets={p.name:sha256(p) for p in OUTPUT.glob('*.jpg')},
        scope='All original Balloon annotations from12admitted TRAIN photos, without blue-pixel filtering. Boxes and labels are unreviewed; source annotation completeness is NOT assumed. Review the full photographs and add missing target boxes or exclude incomplete frames before reporting detector precision/recall. No model inference or automatic positive/background labels.',
        training_approved=False,scene_evaluation_approved=False,test_evaluated=False,deployment_approved=False))
    print(dict(frames=len(frames),annotations=len(samples),queue_sha256=sha256(OUTPUT/'review.json')),flush=True)


if __name__ == '__main__':
    prepare()
