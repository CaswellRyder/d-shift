"""Matched hard-example trial receipts; frozen development is not qualification."""
import argparse
from collections import Counter
from pathlib import Path

from dtr.data import read_json, sha256, write_json
from scripts.balloon_expansion4 import QUEUE, REVIEW, PARENT, MANIFEST, checked_extra, data
from scripts.evaluate_reviewed_balloon_scenes import evaluate
from scripts.summarize_balloon_fragments import PANELS
from scripts.summarize_dense_background_trial import check_baseline
from scripts.summarize_public_balloon_trial import checked_metrics

PROTOCOL = 'configs/balloon-expansion4-hard-trial-20261011.json'
WEIGHTED_PROTOCOL = 'configs/balloon-expansion4-weighted-diagnostic-20261011.json'


def models(weighted=False, convergence=False):
    protocol = read_json(PROTOCOL)
    if convergence and not weighted:
        raise ValueError('Convergence requires the matched weighted experiment')
    if weighted:
        protocol.update(read_json(WEIGHTED_PROTOCOL))
    if convergence:
        protocol.update(read_json('configs/balloon-expansion4-convergence-20261011.json'))
    approved = checked_extra()
    if protocol['parent_sha256'] != sha256(PARENT):
        raise ValueError('Frozen mining parent changed')
    sources, evidence = {'parent':PARENT}, {}
    for mode in ('control','admit'):
        rows, selected = data(mode)
        if [r['id'] for r in selected] != protocol['selected_crop_ids']:
            raise ValueError('Hard mining selection changed')
        identity = [{k:r[k] for k in ('source_image','sha256','label','preserve')} for r in rows]
        for seed in (47,48):
            name = f'{mode}{seed}'
            prefix = ('balloon-expansion4-convergence' if convergence else
                      ('balloon-expansion4-weight16' if weighted else 'balloon-expansion4'))
            run = Path(f'runs/{prefix}-{mode}-seed{seed}-20261011')
            model = run/'student.fp32.tflite'
            p, meta = read_json(run/'provenance.json'), read_json(model.with_suffix('.json'))
            if (p['mode'] != mode or p['seed'] != seed or p['experiment'] != 'expansion4_hard'
                    or p.get('hard_weight',1) != (16 if weighted else 1)
                    or any(p[k] != protocol[k] for k in ('epochs','learning_rate','batch_size'))
                    or p['manifest_sha256'] != sha256(MANIFEST) or p['review_sha256'] != sha256(REVIEW)
                    or p['queue_sha256'] != sha256(Path(QUEUE)/'review.json')
                    or p['data_helper_sha256'] != sha256('scripts/balloon_expansion4.py')
                    or p['parent_model_sha256'] != sha256(PARENT)
                    or p['parent_checkpoint_sha256'] != sha256(PARENT.parent/'color.keras')
                    or p['sample_identity'] != identity or p['ordinary_rows'] != 1487
                    or p['hard_label_rows'] != 20 or p['reviewed_unique_crops'] != 5
                    or p['fixed_final_epoch'] is not True
                    or read_json(run/'status.json')['state'] != 'complete'
                    or meta['provenance_sha256'] != sha256(run/'provenance.json')
                    or meta['sha256'] != sha256(model) or meta['deployment_approved'] is not False
                    or meta['source_student_sha256'] != sha256(run/'student.keras')
                    or not 0 <= meta['max_keras_tflite_delta'] <= .002):
                raise ValueError('Matched trial identity changed')
            sources[name] = model
            evidence[name] = dict(model_sha256=sha256(model), path=str(model),
                provenance_sha256=sha256(run/'provenance.json'), bytes=model.stat().st_size,
                fit=read_json(run/'report.json'))
    return sources, evidence, approved, selected


def feasible(fit):
    return (fit['reviewed_crop_count']==5 and fit['reviewed_crop_correct']>=4
            and fit['ordinary_training_rows']==1487
            and fit['ordinary_training_correct']/fit['ordinary_training_rows']>=.95)


def diagnostic(output):
    if Path(output).exists():
        raise FileExistsError(output)
    _, evidence, approved, selected = models(weighted=True)
    passed = {str(seed):feasible(evidence[f'admit{seed}']['fit']) for seed in (47,48)}
    result = dict(models=evidence,protocol_sha256=sha256(WEIGHTED_PROTOCOL),
        review_sha256=sha256(REVIEW),queue_sha256=sha256(Path(QUEUE)/'review.json'),
        script_sha256=sha256(__file__), reviewed_crops=len(approved),
        selected_ids=[r['id'] for r in selected], training_feasibility=passed,
        both_seeds_pass_training_feasibility=all(passed.values()),
        scope='Training-only weighting diagnostic, not evaluation accuracy.',
        development_evaluated=False,test_evaluated=False,pi_tested=False,
        deployment_approved=False,qualification_achieved=False)
    write_json(output,result)
    print(passed)


def run(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    sources, evidence, approved, selected = models()
    panels = {}
    for panel in PANELS:
        path = Path(f'runs/balloon-expansion4-scenes-{panel}-20261011.json')
        if path.exists():
            raise FileExistsError(path)
        result = evaluate('data/'+panel, 'configs/'+panel+'.json', sources,
            ('mser_confirmed_parts',), 'split_merged', 2.42913421076193)
        write_json(path,result)
        previous = read_json(f'runs/balloon-expansion3-scenes-{panel}-20261011.json')
        metrics = {}
        identities = None
        for name, record in result['results'].items():
            scene = record['full_frame']['mser_confirmed_parts']
            identity = [(r['source'],r['frame_sha256'],r['truth']) for r in scene['frames']]
            if identities is not None and identities != identity:
                raise ValueError('Compared different scenes')
            identities = identity
            metrics[name] = checked_metrics(scene)
            if name=='parent':
                old = previous['results']['parent']['full_frame']['mser_confirmed_parts']['frames']
                for a,b in zip(scene['frames'],old,strict=True):
                    check_baseline(a,b)
        panels[panel] = dict(path=str(path),sha256=sha256(path),frames=len(identities),results=metrics)
        print(panel, metrics, flush=True)
    queue = read_json(Path(QUEUE)/'review.json')
    receipt = dict(models=evidence, panels=panels, protocol_sha256=sha256(PROTOCOL),
        review_sha256=sha256(REVIEW), queue_sha256=sha256(Path(QUEUE)/'review.json'),
        parent_sha256=sha256(PARENT), script_sha256=sha256(__file__),
        reviewed_frames=len(queue['frames']), candidate_crops=len(queue['samples']),
        rejected_frames=len(queue['rejected']), admitted_counts=dict(Counter(r['label'] for r in approved)),
        hard_crop_ids=[r['id'] for r in selected], hard_counts=dict(Counter(r['label'] for r in selected)),
        scope='Assistant-reviewed public TRAIN hard mining with matched controls and fixed epochs; repeated development only.',
        caveats=['New training frames still share environments with earlier public data.',
                 'All 81 approved crops were reviewed; only five low-confidence reviewed crops enter the new hard-label emphasis.',
                 'No blue-positive crop met the fixed hard-mining rule; blue improvement is not implied by training fit.',
                 'No Pi, camera, ESP32 or motor changes in this trial.'],
        test_evaluated=False, pi_tested=False, deployment_approved=False, qualification_achieved=False)
    write_json(output,receipt)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--weighted-diagnostic',action='store_true')
    a = p.parse_args()
    (diagnostic if a.weighted_diagnostic else run)(a.output)
