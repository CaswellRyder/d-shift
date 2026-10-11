"""Verify exact skipped-call parity and bounded development evidence."""
import argparse
from collections import Counter
from pathlib import Path

from dtr.data import read_json,sha256,write_json
from scripts.summarize_balloon_fragments import PANELS
from scripts.summarize_public_balloon_trial import checked_metrics
from scripts.summarize_dense_background_trial import check_baseline


def same_output(a,b):
    for key in ('id','source','truth'):
        if a[key]!=b[key]:
            raise ValueError('Frame identity changed')
    def stable(row):
        return {k:v for k,v in row.items() if k!='latency_ms'}
    if [stable(r) for r in a['detections']]!=[stable(r) for r in b['detections']]:
        raise ValueError('Skipped-call gating changed detections')


def summarize():
    review_path=Path('configs/balloon-expansion3-failure-review-20261011.json')
    review=read_json(review_path)
    queue=read_json(review['queue'])
    if (review['queue_sha256']!=sha256(review['queue']) or review['training_approved'] is not False
            or sorted(r['id'] for r in review['assessments'])!=list(range(len(queue['samples'])))):
        raise ValueError('Diagnostic review changed or incomplete')
    panels={}
    for panel in ('training',)+PANELS:
        training=panel=='training'
        old=Path('runs/balloon-dense-supported-training-v2-20261011.json' if training else
                 f'runs/balloon-dense-supported-{panel}-20261011.json')
        path=Path('runs/balloon-dense-supported-training-gated-20261011.json' if training else
                  f'runs/balloon-dense-supported-gated-{panel}-20261011.json')
        before,after=read_json(old),read_json(path)
        if (after['policy_sha256']!=sha256('scripts/research_dense_supported_selection.py')
                or after['extra_crop_calls']!=0 or after['max_extra_dense_calls_per_frame']!=1
                or after['training_only']!=training or after['deployment_approved'] is not False):
            raise ValueError('Research policy or scope changed')
        for key in ('dense_models','classifier_sha256','queue_sha256','review_sha256'):
            if before[key]!=after[key]:
                raise ValueError('Skipped-call comparison changed inputs')
        for m in after['dense_models'].values():
            if sha256(m['path'])!=m['sha256']:
                raise ValueError('Dense model changed')
        results={}
        for name,result in after['results'].items():
            for a,b in zip(result['frames'],before['results'][name]['frames'],strict=True):
                same_output(a,b)
            if result['dense_calls']!=sum(r['dense_calls'] for r in result['frames']):
                raise ValueError('Dense call count changed')
            results[name]=dict(dense_calls=result['dense_calls'],changed_acceptances=result['changed_acceptances'])
            results[name]['stage_counts' if training else 'metrics']=(result['stage_counts'] if training else checked_metrics(result))
        if not training:
            baseline=read_json(f'runs/balloon-dense-bg-scenes-{panel}-20261011.json')['results']['baseline']
            for a,b in zip(after['results']['baseline']['frames'],baseline['frames'],strict=True):
                check_baseline(a,b)
        panels[panel]=dict(path=str(path),sha256=sha256(path),ungated_path=str(old),ungated_sha256=sha256(old),
            frames=len(after['results']['baseline']['frames']),results=results,exact_gated_detection_parity=True)
    return dict(panels=panels,diagnostic_review_sha256=sha256(review_path),
        diagnostic_queue_sha256=sha256(review['queue']),stage_counts=queue['stage_counts'],
        visual_categories=dict(Counter(r['category'] for r in review['assessments'])),
        policy_sha256=sha256('scripts/research_dense_supported_selection.py'),script_sha256=sha256(__file__),
        decision='Keep as an opt-in research candidate for Pi cost/parity testing, not the active default. Both seeds recover one blue target on the 67-frame panel without extra FPs; other panels unchanged.',
        next_direction='Measure ARMv6 compatibility and gated overhead in isolation before considering integration; retain baseline fallback and require independent camera qualification.',
        caveats=['Training suspects are not automatically false positives or negative labels.',
                 'Development panels overlap and are repeatedly inspected; one-target improvement is weak generalization evidence.',
                 'Skipped inference-call counts are not measured Pi speedups.',
                 'No Pi runtime, camera-inclusive timing or flight qualification in this trial.'],
        test_evaluated=False,deployment_approved=False,pi_tested=False,qualification_achieved=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    write_json(a.output,summarize())
