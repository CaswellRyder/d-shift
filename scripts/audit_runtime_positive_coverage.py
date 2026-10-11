"""TRAIN-only audit of current search crops versus protected training tensors.

Unique >=.7 same-color reviewed annotation matches inherit label candidates,
not approved new labels. No mining of IMG development or reserved test images.
"""
from collections import Counter
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from dtr.data import read_json,sha256,write_json
from dtr.runtime import Predictor
from scripts.train_current_color_clutter import inputs
from scripts.export_current_color_clutter import checked_trial,PARENT
from scripts.prepare_blue_false_alarm_review import source_frames
from scripts.prepare_openimages_balloon_review import sheets
from scripts.build_red_blue_proposal_positives import matched_target
from scripts.balloon_grid_runtime import GridProposals
from scripts.audit_balloon_dense_proposals import supplement
from scripts.research_balloon_search import experimental

QUEUE = Path('data/runtime-positive-coverage-20261011')
OUTPUT = Path('research/balloon-red-blue-20261008/runtime-positive-coverage-20261011.json')


def digest(x):
    return hashlib.sha256(x.tobytes()).hexdigest()


def audit():
    if QUEUE.exists() or OUTPUT.exists():
        raise FileExistsError('Preserve evidence')
    trial = checked_trial()
    rows,labels,ordinary,tensors = inputs('admit')
    known = {}
    for row,tensor,protected in zip(rows,tensors[0],ordinary,strict=True):
        known.setdefault(digest(tensor),[]).append(dict(label=row['label'],protected=bool(protected)))
    paths = {'parent':PARENT,**{k:Path(v['path'])/'student.fp32.tflite' for k,v in trial['trials'].items()}}
    models,identities = {},{}
    for name,path in paths.items():
        models[name] = Predictor(path,allow_unvalidated=True)
        meta = read_json(path.with_suffix('.json'))
        if meta['threshold']!=.8 or meta['sha256']!=sha256(path):
            raise ValueError('Model operating point changed')
        identities[name] = dict(path=str(path),sha256=sha256(path),metadata_sha256=sha256(path.with_suffix('.json')))
    grid_path = Path('runs/balloon-grid-convergence-coarse-seed47-20261011/proposal.fp32.tflite')
    grid = GridProposals(grid_path)
    QUEUE.mkdir(parents=True)
    (QUEUE/'crops').mkdir()
    (QUEUE/'frames').mkdir()
    samples,frames,seen = [],[],set()
    for source,rgb in source_frames():
        if source['origin']!='complete_fit':
            continue
        targets = [dict(t,scan_box=t['box'],annotation_id=i) for i,t in enumerate(source['truth'])]
        base = experimental(rgb,'mser')
        candidates = supplement(base,grid.predict(rgb))
        fid = len(frames)
        path = f'frames/{fid:03d}.png'
        Image.fromarray(rgb).save(QUEUE/path)
        frames.append(dict(source,id=fid,path=path,sha256=sha256(QUEUE/path)))
        for c in candidates:
            target = matched_target(c,targets)
            if target is None:
                continue
            a,b,d,e = c['crop_box']
            crop = rgb[b:e,a:d]
            tensor = np.asarray(Image.fromarray(crop).resize((64,64),Image.Resampling.BILINEAR),np.float32)
            h = digest(tensor)
            key = source['source'],h
            if key in seen:
                continue
            seen.add(key)
            predictions = {name:{k:v for k,v in model.predict(crop).items() if k!='latency_ms'} for name,model in models.items()}
            path = f'crops/{len(samples):03d}.png'
            Image.fromarray(crop).save(QUEUE/path)
            labels_in_training = known.get(h,[])
            samples.append(dict(id=len(samples),frame_id=fid,source=source['source'],path=path,sha256=sha256(QUEUE/path),
                split='train',label=target['label'],label_status='inherited_candidate_requires_review',
                proposal=c,target=target,tensor_sha256=h,in_training=bool(labels_in_training),
                protected_same_label=any(r['protected'] and r['label']==target['label'] for r in labels_in_training),
                predictions=predictions))
    if len(frames)!=64:
        raise ValueError('Require64 complete FIT scenes')
    results = {}
    for name in models:
        results[name] = {}
        for color in ('red_balloon','blue_balloon'):
            selected = [s for s in samples if s['label']==color]
            def accepted(r,n):
                return r['predictions'][n]['label']==color and r['predictions'][n]['accepted']
            results[name][color] = dict(total=len(selected),accepted=sum(accepted(r,name) for r in selected),
                parent_accepted=sum(accepted(r,'parent') for r in selected),
                lost_parent_ids=[r['id'] for r in selected if accepted(r,'parent') and not accepted(r,name)],
                missing_protected_tensor=sum(not r['protected_same_label'] for r in selected))
    sheets(QUEUE,'crops',samples,6,(160,160))
    write_json(QUEUE/'review.json',dict(samples=samples,frames=frames,classifiers=identities,
        search=dict(path=str(grid_path),sha256=sha256(grid_path),metadata_sha256=sha256(grid_path.with_suffix('.json'))),
        script_sha256=sha256(__file__),helpers_sha256={p:sha256(p) for p in (
            'scripts/build_red_blue_proposal_positives.py','scripts/prepare_blue_false_alarm_review.py',
            'scripts/research_balloon_search.py','scripts/audit_balloon_dense_proposals.py')},
        sheets={p.name:sha256(p) for p in sorted(QUEUE.glob('*.jpg'))},
        training_approved=False,review_required=True,test_evaluated=False,deployment_approved=False))
    write_json(OUTPUT,dict(results=results,queue=dict(path=str(QUEUE),sha256=sha256(QUEUE/'review.json')),
        frames=64,unique_crops=len(samples),counts=dict(Counter(r['label'] for r in samples)),
        script_sha256=sha256(__file__),
        scope='Exact resized-tensor coverage, not semantic data novelty or independent accuracy. Labels inherited through strict geometry, pending visual review. No retraining yet.',
        test_evaluated=False,pi_tested=False,deployment_approved=False,qualification_achieved=False))
    print(dict(results=results,crops=len(samples)))


if __name__=='__main__':
    audit()
