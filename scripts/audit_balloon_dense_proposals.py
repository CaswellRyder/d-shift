"""Frozen proposal decoder and sparse-training audit, not scene precision."""
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, write_json
from dtr.tracking import iou
from scripts.evaluate_balloon_edge_search import coverage
from scripts.research_balloon_search import experimental
from scripts.train_balloon_dense_proposals import checked_data


def decode(values):
    if values.shape != (15,20,6) or not np.isfinite(values).all():
        raise ValueError('Require finite dense proposal output')
    heat = 1/(1+np.exp(-np.clip(values[:,:,:2],-80,80)))
    pool = []
    for group in (0,1):
        peaks = (heat[:,:,group] >= .1)&(heat[:,:,group] == cv2.dilate(heat[:,:,group],np.ones((3,3),np.uint8)))
        for y,x in np.argwhere(peaks):
            ox,oy,bw,bh = values[y,x,2:]
            if not np.all((values[y,x,2:] >= 0)&(values[y,x,2:] <= 1)):
                raise ValueError('Box channels must be normalized')
            cx,cy = (x+ox)*16,(y+oy)*16
            w,h = bw*320,bh*240
            a,b,c,d = max(0.,cx-w/2),max(0.,cy-h/2),min(320.,cx+w/2),min(240.,cy+h/2)
            if min(c-a,d-b) < 4:
                continue
            pad = max(c-a,d-b)*.12
            pool.append(dict(box=[float(a),float(b),float(c),float(d)],
                crop_box=[max(0,int(np.floor(a-pad))),max(0,int(np.floor(b-pad))),
                          min(320,int(np.ceil(c+pad))),min(240,int(np.ceil(d+pad)))],
                area=float((c-a)*(d-b)),color_group=group,proposal_score=float(heat[y,x,group])))
    found = []
    for row in sorted(pool,key=lambda r:-r['proposal_score']):
        if not any(r['color_group']==row['color_group'] and iou(r['box'],row['box'])>.65 for r in found):
            found.append(row)
        if len(found)==12:
            break
    return found


def supplement(base, learned):
    found = list(base)
    for row in learned:
        if len(found)>=min(12,len(base)+2):
            break
        if row['color_group']==1 and not any(iou(row['box'],r['box'])>.65 for r in found):
            found.append(row)
    return found


def audit(root,run):
    root,run = Path(root),Path(run)
    doc,_,_,folds,_ = checked_data(root)
    provenance,report = read_json(run/'provenance.json'),read_json(run/'report.json')
    if (provenance['manifest_sha256'] != sha256(root/'manifest.json')
            or provenance['folds'] != folds.tolist() or provenance['epochs']!=80
            or read_json(run/'status.json')['state']!='complete'
            or report['provenance_sha256'] != sha256(run/'provenance.json')
            or report['model_sha256'] != sha256(run/'proposal.keras')
            or report['predictions_sha256'] != sha256(run/'predictions.npy')):
        raise ValueError('Training identity changed')
    values = np.load(run/'predictions.npy',allow_pickle=False)
    if values.shape != (len(doc['frames']),15,20,6):
        raise ValueError('Incomplete predictions')
    frames = []
    for row,value,fold in zip(doc['frames'],values,folds,strict=True):
        with Image.open(root/row['path']) as image:
            rgb = np.asarray(image.convert('RGB'))
        baseline,learned = experimental(rgb,'mser'),decode(value)
        added = supplement(baseline,learned)
        frames.append(dict(source=row['source'],frame_sha256=row['sha256'],fold=int(fold),
            truth=row['truth'],coverage={k:coverage(row['truth'],p) for k,p in
                (('baseline',baseline),('learned',learned),('supplement',added))},
            learned_proposals=learned,extra_calls=len(added)-len(baseline)))
    totals = {}
    for split in ('fit','heldout_groups'):
        selected = [r for r in frames if (r['fold']==0)==(split=='heldout_groups')]
        totals[split] = {label:dict(targets=sum(r['coverage']['baseline'][label]['targets'] for r in selected),
            **{k:sum(r['coverage'][k][label]['localized'] for r in selected)
               for k in ('baseline','learned','supplement')}) for label in ('red_balloon','blue_balloon')}
    return dict(frames=frames,totals=totals,seed=provenance['seed'],parameters=provenance['parameters'],
        model_sha256=report['model_sha256'],provenance_sha256=sha256(run/'provenance.json'),
        data_manifest_sha256=sha256(root/'manifest.json'),script_sha256=sha256(__file__),
        proposal_threshold=.1,candidate_limit=12,supplement_limit=2,
        scope='One-to-one known TRAIN target coverage. Heldout groups excluded only from localizer fit; related environments and parent classifier are not independently held out.',
        classifier_evaluated=False,test_evaluated=False,pi_tested=False,deployment_approved=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('data','run','output'):
        p.add_argument('--'+arg,required=True)
    a=p.parse_args()
    if Path(a.output).exists():
        raise FileExistsError(a.output)
    result=audit(a.data,a.run)
    write_json(a.output,result)
    print(result['totals'])
