"""Training-only stage attribution and unlabelled error-review crops.

Incomplete scene annotations cannot establish false positives. Suspect accepted
crops are queued for visual review, never automatically called background.
"""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw,ImageOps

from dtr.data import read_json,sha256,write_json
from dtr.runtime import Predictor
from dtr.tracking import iou
from scripts.balloon_expansion3 import QUEUE,REVIEW,BOOTSTRAP,checked_extra
from scripts.audit_balloon_search_stages import indoor_frames
from scripts.research_balloon_search import experimental
from scripts.research_predictor import classify_candidates
from scripts.pi_balloon_search_bench import select_detections
from scripts.train_balloon_reviewed_clutter import PARENT


def explain(target,found):
    matching = [r for r in found if iou(target['box'],r['box'])>=.5]
    raw = [r for r in matching if r['raw_accepted'] and r['label']==target['label']]
    selected = [r for r in raw if r['accepted']]
    return ('covered' if selected else 'selection' if raw else
            'classification' if matching else 'proposal_localization')


def suspect(found,truth):
    # Wrong color, duplicate boxes and missing annotations all need review.
    return [r for r in found if r['accepted'] and not any(
        t['label']==r['label'] and iou(t['box'],r['box'])>=.5 for t in truth)]


def audit(output):
    output=Path(output)
    if output.exists():
        raise FileExistsError(output)
    approved=checked_extra()
    allowed={r['source_image']:r for r in approved}
    parent=read_json(PARENT/'color.fp32.json')
    model=PARENT/'color.fp32.tflite'
    if parent['sha256']!=sha256(model) or parent['threshold']!=.8:
        raise ValueError('Current research candidate changed')
    predictor=Predictor(model,allow_unvalidated=True)
    sources=indoor_frames(QUEUE,REVIEW,BOOTSTRAP,'data/raw/engdes2-red-blue-v1-20261008/dataset.zip')
    output.mkdir(parents=True)
    (output/'crops').mkdir()
    (output/'frames').mkdir()
    frames,rows=[],[]
    for source,rgb,truth in sources:
        if rgb.shape!=(240,320,3) or rgb.dtype!=np.uint8:
            raise ValueError('Require canonical RGB training frames')
        source='train/'+source
        if source not in allowed:
            raise ValueError('Source is not approved TRAIN')
        ident=len(frames)
        found=select_detections(classify_candidates(predictor,rgb,experimental(rgb,'mser')),
                                'split_merged',2.42913421076193)
        path=output/f'frames/{ident:03d}.png'
        Image.fromarray(rgb).save(path)
        targets=[dict(t,stage=explain(t,found)) for t in truth]
        frame=dict(id=ident,path=str(path.relative_to(output)),sha256=sha256(path),
            source_image=source,source_sha256=allowed[source]['source_sha256'],truth=truth,
            targets=targets,detections=found)
        frames.append(frame)
        for candidate in suspect(found,truth):
            x,y,z,w=candidate['crop_box']
            crop=output/f'crops/{len(rows):03d}.png'
            Image.fromarray(rgb[y:w,x:z]).save(crop)
            rows.append(dict(id=len(rows),frame_id=ident,source_image=source,
                source_sha256=frame['source_sha256'],path=str(crop.relative_to(output)),
                sha256=sha256(crop),crop_box=candidate['crop_box'],box=candidate['box'],
                label='unreviewed',prediction={k:candidate[k] for k in ('label','score','scores')},
                best_known_iou=max((iou(candidate['box'],t['box']) for t in truth),default=0.)))
    for kind,items,columns,per_page,tile in (
            ('crops',rows,6,24,(160,160)),('frames',frames,3,9,(320,270))):
        for start in range(0,len(items),per_page):
            sheet=Image.new('RGB',(columns*tile[0],((per_page+columns-1)//columns)*tile[1]),'white')
            draw=ImageDraw.Draw(sheet)
            for n,row in enumerate(items[start:start+per_page]):
                x,y=n%columns*tile[0],n//columns*tile[1]
                with Image.open(output/row['path']) as image:
                    sheet.paste(ImageOps.contain(image,(tile[0],tile[1]-30)),(x,y+30))
                draw.text((x+2,y+2),f"id {row['id']}"+(f" frame {row['frame_id']}" if kind=='crops' else ''),fill='black')
            sheet.save(output/f'{kind}-{start:03d}.png')
    counts={label:dict(Counter(t['stage'] for f in frames for t in f['targets'] if t['label']==label))
            for label in ('red_balloon','blue_balloon')}
    report=dict(frames=frames,samples=rows,stage_counts=counts,model_sha256=sha256(model),
        model_metadata_sha256=sha256(model.with_suffix('.json')),parent_queue_sha256=sha256(Path(QUEUE)/'review.json'),
        parent_review_sha256=sha256(REVIEW),script_sha256=sha256(__file__),
        helper_hashes={p:sha256(p) for p in ('scripts/balloon_expansion3.py','scripts/research_balloon_search.py',
            'scripts/research_predictor.py','scripts/pi_balloon_search_bench.py')},
        scope='Approved TRAIN only; per-target attribution not one-to-one scene metrics. Suspect acceptances remain unlabelled; missing annotations are not background.',
        training_approved=False,review_required=True,test_evaluated=False,deployment_approved=False)
    write_json(output/'review.json',report)
    return dict(frames=len(frames),suspect_crops=len(rows),stage_counts=counts,
                queue_sha256=sha256(output/'review.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    print(audit(p.parse_args().output))
