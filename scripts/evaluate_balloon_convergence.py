"""Verify training convergence before frozen final-checkpoint development replay."""
import argparse
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from dtr.runtime import Predictor
from scripts.balloon_expansion4 import QUEUE, CLASSES
from scripts.evaluate_balloon_expansion4 import models, feasible
from scripts.evaluate_reviewed_balloon_scenes import evaluate
from scripts.summarize_balloon_fragments import PANELS
from scripts.summarize_dense_background_trial import check_baseline
from scripts.summarize_public_balloon_trial import checked_metrics

PROTOCOL = 'configs/balloon-expansion4-convergence-20261011.json'


def check_trace(trace, fit, labels):
    if [r['epoch'] for r in trace['records']] != [0,1,10,20,30,40,50,60,70,80]:
        raise ValueError('Incomplete fixed training trajectory')
    for row in trace['records']:
        values = np.asarray(row['hard_probabilities'])
        if (row['ordinary_rows'] != 1487 or not 0<=row['ordinary_correct']<=1487
                or row['hard_rows'] != 5 or values.shape != (5,3) or not np.isfinite(values).all()
                or np.any(values<0) or np.any(values>1)
                or not np.allclose(values.sum(1),1,atol=1e-6,rtol=0)
                or row['hard_correct'] != int((values.argmax(1)==labels).sum())
                or not np.allclose(row['reviewed_class_probabilities'], values[np.arange(5),labels],atol=1e-7,rtol=0)):
            raise ValueError('Invalid training trajectory')
    last = trace['records'][-1]
    if (last['hard_correct'] != fit['reviewed_crop_correct']
            or last['ordinary_correct'] != fit['ordinary_training_correct']):
        raise ValueError('Final fit and trajectory disagree')


def diagnose(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    sources, evidence, _, selected = models(weighted=True,convergence=True)
    protocol = read_json(PROTOCOL)
    labels = np.array([CLASSES.index(r['label']) for r in selected])
    for name,item in evidence.items():
        root = Path(item['path']).parent
        p = read_json(root/'provenance.json')
        trace = read_json(root/'training_trace.json')
        if (p['convergence_protocol_sha256'] != sha256(PROTOCOL)
                or p['trace_helper_sha256'] != sha256('scripts/balloon_convergence.py')
                or item['fit']['training_trace_sha256'] != sha256(root/'training_trace.json')
                or trace['hard_ids'] != protocol['selected_crop_ids']):
            raise ValueError('Training trajectory identity changed')
        check_trace(trace,item['fit'],labels)
        predictor = Predictor(sources[name],allow_unvalidated=True)
        actual = np.array([predictor.predict(Path(QUEUE)/r['path'])['scores'] for r in selected])
        expected = np.asarray(trace['records'][-1]['hard_probabilities'])
        if np.max(np.abs(actual-expected)) > .002:
            raise ValueError('Export does not reproduce final training probe')
        item.update(training_trace_sha256=sha256(root/'training_trace.json'),
                    trajectory=trace['records'],training_feasible=feasible(item['fit']))
    both = all(evidence[f'admit{s}']['training_feasible'] for s in (47,48))
    receipt = dict(models=evidence, protocol_sha256=sha256(PROTOCOL),
        parent_sha256=sha256(sources['parent']), both_admit_seeds_training_feasible=both,
        script_sha256=sha256(__file__),scope='TRAIN convergence and retention only, not held-out accuracy',
        development_evaluated=False,test_evaluated=False,pi_tested=False,
        deployment_approved=False,qualification_achieved=False)
    write_json(output,receipt)
    print({n:(r['fit']['reviewed_crop_correct'],r['fit']['ordinary_training_correct']) for n,r in evidence.items()})
    return receipt


def development(diagnostic, output):
    diagnostic, output = Path(diagnostic), Path(output)
    if output.exists():
        raise FileExistsError(output)
    audit = read_json(diagnostic)
    if audit['both_admit_seeds_training_feasible'] is not True or audit['protocol_sha256'] != sha256(PROTOCOL):
        raise ValueError('Both seeds must meet training-only feasibility before evaluation')
    sources, evidence, _, _ = models(weighted=True,convergence=True)
    for name,row in evidence.items():
        if (row['model_sha256'] != audit['models'][name]['model_sha256']
                or row['provenance_sha256'] != audit['models'][name]['provenance_sha256']
                or (name.startswith('admit') and not feasible(row['fit']))):
            raise ValueError('Training feasibility or model identity changed')
    panels = {}
    for panel in PANELS:
        path = Path(f'runs/balloon-expansion4-convergence-scenes-{panel}-20261011.json')
        if path.exists():
            raise FileExistsError(path)
        result = evaluate('data/'+panel,'configs/'+panel+'.json',sources,
                          ('mser_confirmed_parts',),'split_merged',2.42913421076193)
        write_json(path,result)
        prior = read_json(f'runs/balloon-expansion4-scenes-{panel}-20261011.json')
        metrics = {}
        for name,record in result['results'].items():
            scenes = record['full_frame']['mser_confirmed_parts']
            metrics[name] = checked_metrics(scenes)
            if name=='parent':
                old = prior['results']['parent']['full_frame']['mser_confirmed_parts']['frames']
                for a,b in zip(scenes['frames'],old,strict=True):
                    check_baseline(a,b)
        panels[panel] = dict(path=str(path),sha256=sha256(path),results=metrics)
        print(panel,metrics,flush=True)
    write_json(output,dict(diagnostic_sha256=sha256(diagnostic),panels=panels,
        protocol_sha256=sha256(PROTOCOL),script_sha256=sha256(__file__),
        scope='Repeated development, not independent qualification; final epoch only',
        test_evaluated=False,pi_tested=False,deployment_approved=False,qualification_achieved=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--development-from')
    a = p.parse_args()
    if a.development_from:
        development(a.development_from,a.output)
    else:
        diagnose(a.output)
