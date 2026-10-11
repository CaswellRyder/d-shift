"""Verify public localizer training, shared original holdout and scene evidence."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from dtr.data import read_json,sha256,write_json
from scripts.audit_balloon_dense_proposals import audit
from scripts.balloon_dense_public_data import PROTOCOL
from scripts.train_balloon_dense_proposals import checked_data
from scripts.summarize_balloon_fragments import PANELS
from scripts.summarize_dense_background_trial import check_baseline
from scripts.summarize_public_balloon_trial import checked_metrics


def split_coverage(frames, base_count):
    result={}
    for name,selected in (
        ('common_fit',[r for i,r in enumerate(frames) if i<base_count and r['fold']!=0]),
        ('common_grouped_holdout',[r for i,r in enumerate(frames) if i<base_count and r['fold']==0]),
        ('extra_fit',frames[base_count:])):
        result[name]={label:dict(targets=sum(r['coverage']['baseline'][label]['targets'] for r in selected),
            **{k:sum(r['coverage'][k][label]['localized'] for r in selected)
               for k in ('baseline','learned','supplement')}) for label in ('red_balloon','blue_balloon')}
    return result


def training():
    protocol=read_json(PROTOCOL)
    base,x,y,folds,groups=checked_data(protocol['base_data'])
    trials={}
    datasets={}
    for mode in ('control','admit'):
        root=Path(f'data/balloon-dense-public-{mode}-20261011')
        doc,xx,yy,ff,gg=checked_data(root)
        n=len(base['frames'])
        if (not np.array_equal(xx[:n],x) or not np.array_equal(yy[:n],y)
                or not np.array_equal(ff[:n],folds) or gg[:n]!=groups or np.any(ff[n:]==0)):
            raise ValueError('Original pixels, targets, or grouped holdout changed')
        datasets[mode]=dict(path=str(root),sha256=sha256(root/'manifest.json'),
            original_frames=n,extra_frames=len(doc['frames'])-n,original_holdout_unchanged=True,
            extra_box_counts=dict(Counter(t['label'] for r in doc['frames'][n:] for t in r['truth'])),
            extra_heat_cells=int(yy[n:,:,:,2:4].sum()))
        trials[mode]={}
        for seed in (47,48):
            run=Path(f'runs/balloon-dense-public-{mode}-seed{seed}-20261011')
            coverage=audit(root,run)
            p,report=read_json(run/'provenance.json'),read_json(run/'report.json')
            meta=read_json(run/'proposal.fp32.json')
            if (any(p[k]!=protocol[k] for k in ('epochs','batch_size','learning_rate','box_objective'))
                    or p['seed']!=seed or p['parameters']!=2342 or p['giou_weight']!=2
                    or p['fixed_final_epoch'] is not True
                    or p['script_sha256']!=sha256('scripts/train_balloon_dense_proposals.py')
                    or meta['sha256']!=sha256(run/'proposal.fp32.tflite')
                    or meta['manifest_sha256']!=sha256(root/'manifest.json')
                    or meta['source_checkpoint_sha256']!=sha256(run/'proposal.keras')
                    or meta['provenance_sha256']!=sha256(run/'provenance.json')
                    or meta['decoder_sha256']!=sha256('scripts/audit_balloon_dense_proposals.py')
                    or meta['decoded_parity_frames']!=len(doc['frames'])
                    or not 0<=meta['max_tensor_delta']<=.002
                    or meta['parameters']!=2342 or meta['deployment_approved'] is not False):
                raise ValueError('Training or export protocol changed')
            trials[mode][str(seed)]=dict(path=str(run),provenance_sha256=sha256(run/'provenance.json'),
                model_sha256=meta['sha256'],metadata_sha256=sha256(run/'proposal.fp32.json'),
                fit=report,export=meta,coverage=split_coverage(coverage['frames'],n))
    if datasets['control']['extra_frames']!=datasets['admit']['extra_frames']:
        raise ValueError('Sample-count control changed')
    return trials,datasets


def summarize(output):
    output=Path(output)
    if output.exists():
        raise FileExistsError(output)
    trials,datasets=training()
    panels={}
    for panel in PANELS:
        compared={}
        baseline=None
        for mode in ('control','admit'):
            path=Path(f'runs/balloon-dense-public-{mode}-scenes-{panel}-20261011.json')
            doc=read_json(path)
            if (doc['classifier_threshold']!=.8 or doc['matching_iou']!=.5
                    or doc['selection']!='split_merged' or doc['proposal_threshold']!=.1
                    or doc['split_max_aspect']!=2.42913421076193):
                raise ValueError('Development operating point changed')
            if baseline is None:
                old=read_json(f'runs/balloon-dense-overlap-scenes-{panel}-20261011.json')
                baseline=old['results']['baseline']['frames']
            for a,b in zip(doc['results']['baseline']['frames'],baseline,strict=True):
                check_baseline(a,b)
            for seed,identity in doc['dense_models'].items():
                expected=trials[mode][seed]
                if (identity['sha256']!=expected['model_sha256']
                        or identity['metadata_sha256']!=expected['metadata_sha256']):
                    raise ValueError('Scene model changed')
            compared[mode]=dict(path=str(path),sha256=sha256(path),
                results={k:checked_metrics(v) for k,v in doc['results'].items()})
        panels[panel]=compared
    write_json(output,dict(trials=trials,datasets=datasets,panels=panels,
        protocol_sha256=sha256(PROTOCOL),script_sha256=sha256(__file__),
        scope='Sample-count matched TRAIN expansion; shared known-environment grouped holdout and repeated development, not independent qualification',
        caveats=['Controls match frame count and optimizer steps, not annotation counts or background-mask density.',
                 'Unknown regions in public scenes are ignored, not negative labels.',
                 'No classifier changes, threshold sweep, Pi timing or camera-inclusive verification.'],
        test_evaluated=False,pi_tested=False,deployment_approved=False,qualification_achieved=False))
    print({p:{m:v['results'] for m,v in arms.items()} for p,arms in panels.items()})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    summarize(p.parse_args().output)
