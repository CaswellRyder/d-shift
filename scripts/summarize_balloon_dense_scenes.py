"""Verify scene evidence and keep failed dense-supplement research off the Pi."""
import argparse
from collections import Counter
from pathlib import Path

from dtr.data import read_json,sha256,write_json
from dtr.tracking import iou
from scripts.summarize_balloon_fragments import PANELS
from scripts.summarize_public_balloon_trial import checked_metrics
from scripts.evaluate_balloon_dense_scenes import CLASSIFIER


def added_overlap(baseline,frames):
    counts=Counter()
    for old,new in zip(baseline,frames,strict=True):
        if any(old[k]!=new[k] for k in ('id','source','frame_sha256','truth')):
            raise ValueError('Compared frame identity changed')
        size=len(old['detections'])
        if not size<=len(new['detections'])<=min(12,size+2):
            raise ValueError('Added crop budget changed')
        for a,b in zip(old['detections'],new['detections'][:size],strict=True):
            if any(a[k]!=b[k] for k in ('box','crop_box','label','scores','raw_accepted')):
                raise ValueError('Baseline proposals or classifier output changed')
        for row in new['detections'][size:]:
            if not row['accepted']:
                continue
            overlap=max((iou(row['box'],t['box']) for t in new['truth'] if t['label']==row['label']),default=0.)
            counts['iou_ge_0.5' if overlap>=.5 else 'iou_0.1_to_0.5' if overlap>=.1 else 'iou_lt_0.1']+=1
    return dict(counts)


def summarize():
    models={}
    for seed in (47,48):
        path=Path(f'runs/balloon-dense-proposals-seed{seed}-20261011/proposal.fp32.tflite')
        meta=read_json(path.with_suffix('.json'))
        if (meta['sha256']!=sha256(path) or meta['parameters']!=2342
                or meta['decoded_parity_frames']!=108 or not 0<=meta['max_tensor_delta']<=.002
                or meta['deployment_approved'] is not False or meta['pi_tested'] is not False):
            raise ValueError('Dense export identity changed')
        models[str(seed)]=dict(path=str(path),metadata_sha256=sha256(path.with_suffix('.json')),**meta)
    panels={}
    for panel in PANELS:
        path=Path(f'runs/balloon-dense-scenes-{panel}-20261011.json')
        doc=read_json(path)
        if (doc['classifier_sha256']!=sha256(CLASSIFIER) or doc['selection']!='split_merged'
                or doc['split_max_aspect']!=2.42913421076193 or doc['classifier_threshold']!=.8
                or doc['proposal_threshold']!=.1 or doc['matching_iou']!=.5
                or doc['test_evaluated'] is not False or doc['deployment_approved'] is not False
                or set(doc['results'])!={'baseline','dense47','dense48'}):
            raise ValueError('Frozen comparison scope changed')
        for seed,model in doc['dense_models'].items():
            if model['sha256']!=models[seed]['sha256'] or model['metadata_sha256']!=models[seed]['metadata_sha256']:
                raise ValueError('Evaluated model changed')
        baseline=doc['results']['baseline']
        previous=read_json(f'runs/balloon-fragment-color-{panel}-20261011.json')
        if checked_metrics(baseline)!=checked_metrics(previous['results']['color47']['full_frame']['mser_confirmed_parts']):
            raise ValueError('Baseline differs from prior scene comparison')
        results={name:checked_metrics(result) for name,result in doc['results'].items()}
        histograms={name:added_overlap(baseline['frames'],doc['results'][name]['frames']) for name in ('dense47','dense48')}
        panels[panel]=dict(path=str(path),sha256=sha256(path),frames=len(baseline['frames']),results=results,
            accepted_addition_overlap=histograms,queue_sha256=doc['queue_sha256'],review_sha256=doc['review_sha256'])
    return dict(models=models,panels=panels,classifier_sha256=sha256(CLASSIFIER),summary_script_sha256=sha256(__file__),
        decision='Do not promote either dense supplement: blue precision regresses on every development panel. Keep earlier classifier-only integration candidate.',
        next_direction='Improve proposal box supervision and reviewed training background coverage, not thresholds on these already seen panels.',
        caveats=['Repeated development panels are not independent qualification.',
                 'Overlap histogram counts accepted additions, not disjoint TP/FP categories; overlapping proposals may duplicate an existing detection.',
                 'Small FP32 exports are conversion evidence only; no Pi runtime or camera-inclusive timing measured.'],
        test_evaluated=False,deployment_approved=False,pi_tested=False,qualification_achieved=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    write_json(a.output,summarize())
