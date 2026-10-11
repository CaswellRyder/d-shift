"""Inspect saved TRAIN-center predictions; oracle centers are not detections."""
import argparse
from pathlib import Path

import numpy as np

from dtr.data import read_json,sha256,write_json
from dtr.tracking import iou
from scripts.train_balloon_dense_proposals import checked_data


def diagnose(output):
    output=Path(output)
    if output.exists():
        raise FileExistsError(output)
    root=Path('data/balloon-dense-public-admit-20261011')
    doc,_,targets,folds,_=checked_data(root)
    results={}
    for seed in (47,48):
        run=Path(f'runs/balloon-dense-public-admit-seed{seed}-20261011')
        report,p=read_json(run/'report.json'),read_json(run/'provenance.json')
        if (p['manifest_sha256']!=sha256(root/'manifest.json')
                or report['predictions_sha256']!=sha256(run/'predictions.npy')
                or report['model_sha256']!=sha256(run/'proposal.keras')):
            raise ValueError('Training prediction identity changed')
        values=np.load(run/'predictions.npy',allow_pickle=False)
        rows=[]
        for index,(frame,fold) in enumerate(zip(doc['frames'],folds,strict=True)):
            if fold==0 or frame.get('expansion_kind')!='public':
                continue
            for truth in frame['truth']:
                a,b,c,d=truth['box']
                cx,cy=(a+c)/2,(b+d)/2
                gx,gy=min(19,int(cx/16)),min(14,int(cy/16))
                collision=not bool(targets[index,gy,gx,8])
                heat=1/(1+np.exp(-np.clip(values[index,gy,gx,:2],-80,80)))
                ox,oy,bw,bh=values[index,gy,gx,2:]
                px,py=(gx+ox)*16,(gy+oy)*16
                width,height=bw*320,bh*240
                box=[max(0,float(px-width/2)),max(0,float(py-height/2)),
                     min(320,float(px+width/2)),min(240,float(py+height/2))]
                color=0 if truth['label']=='red_balloon' else 1
                rows.append(dict(source=frame['source'],truth=truth,box_supervision_collision=collision,
                    true_center_heat=float(heat[color]),predicted_box_at_true_cell=box,
                    true_cell_box_iou=float(iou(box,truth['box'])),
                    width_ratio=float(width/(c-a)),height_ratio=float(height/(d-b))))
        stats={}
        for label in ('red_balloon','blue_balloon'):
            selected=[r for r in rows if r['truth']['label']==label and not r['box_supervision_collision']]
            stats[label]=dict(noncollision_targets=len(selected),
                center_heat_below_decoder_threshold=sum(r['true_center_heat']<.1 for r in selected),
                true_cell_box_iou_ge_half=sum(r['true_cell_box_iou']>=.5 for r in selected),
                median_iou=float(np.median([r['true_cell_box_iou'] for r in selected])),
                median_width_ratio=float(np.median([r['width_ratio'] for r in selected])),
                median_height_ratio=float(np.median([r['height_ratio'] for r in selected])))
        results[str(seed)]=dict(rows=rows,statistics=stats,predictions_sha256=report['predictions_sha256'])
    write_json(output,dict(results=results,manifest_sha256=sha256(root/'manifest.json'),
        script_sha256=sha256(__file__),scope='TRAIN oracle-center geometry diagnostic, not detection or qualification metrics',
        development_evaluated=False,test_evaluated=False,deployment_approved=False))
    print({s:r['statistics'] for s,r in results.items()})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    diagnose(p.parse_args().output)
