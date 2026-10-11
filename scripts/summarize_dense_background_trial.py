"""Matched background-supervision trial receipts; no model promotion."""
import argparse
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from scripts.audit_balloon_dense_proposals import decode
from scripts.balloon_dense_complete import validate_derivative
from scripts.evaluate_balloon_dense_scenes import CLASSIFIER
from scripts.evaluate_red_blue_development import LABELS, detection_counts, metrics
from scripts.summarize_balloon_dense_scenes import added_overlap
from scripts.summarize_balloon_fragments import PANELS
from scripts.summarize_public_balloon_trial import checked_metrics
from scripts.train_balloon_dense_proposals import checked_data

SPARSE = Path('data/balloon-dense-proposal-training-20261011')
COMPLETE = Path('data/balloon-dense-complete-training-20261011')


def check_baseline(a,b):
    if any(a[key]!=b[key] for key in ('id','source','frame_sha256','truth','counts')):
        raise ValueError('Baseline scene identity changed')
    def stable(rows):
        return [{k:v for k,v in r.items() if k!='latency_ms'} for r in rows]
    if stable(a['detections'])!=stable(b['detections']):
        raise ValueError('Baseline scene output changed')


def summarize():
    old,x,y,folds,groups = checked_data(SPARSE)
    new,cx,cy,cf,cg = checked_data(COMPLETE)
    approved = validate_derivative(new)
    if (not np.array_equal(x,cx) or not np.array_equal(folds,cf) or groups!=cg
            or not np.array_equal(y[folds==0],cy[folds==0])):
        raise ValueError('Matched data or grouped holdout changed')
    trials = {}
    for mode,root in (('sparse',SPARSE),('complete',COMPLETE)):
        trials[mode] = {}
        for seed in (47,48):
            run = Path(f'runs/balloon-dense-bg-{mode}-seed{seed}-20261011')
            provenance,report = read_json(run/'provenance.json'),read_json(run/'report.json')
            coverage = read_json(run/'coverage.json')
            if (read_json(run/'status.json')['state']!='complete'
                    or provenance['manifest_sha256']!=sha256(root/'manifest.json')
                    or provenance['seed']!=seed or provenance['epochs']!=80
                    or provenance['folds']!=folds.tolist() or provenance['groups']!=groups
                    or report['provenance_sha256']!=sha256(run/'provenance.json')
                    or report['model_sha256']!=sha256(run/'proposal.keras')
                    or report['predictions_sha256']!=sha256(run/'predictions.npy')
                    or coverage['model_sha256']!=report['model_sha256']):
                raise ValueError('Training identity changed')
            values = np.load(run/'predictions.npy',allow_pickle=False)
            counts = {label:dict(tp=0,fp=0,fn=0) for label in LABELS}
            for i in sorted(approved):
                detections = [dict(p,label=LABELS[p['color_group']],accepted=True,
                                   score=p['proposal_score']) for p in decode(values[i])]
                c = detection_counts(old['frames'][i]['truth'],detections)
                for label in LABELS:
                    for key in counts[label]:
                        counts[label][key] += c[label][key]
            item = dict(run=str(run),**report,coverage=coverage['totals'],
                coverage_sha256=sha256(run/'coverage.json'),complete_fit_proposals=metrics(counts))
            if mode=='sparse':
                original = Path(f'runs/balloon-dense-proposals-seed{seed}-20261011/predictions.npy')
                if not np.array_equal(values,np.load(original,allow_pickle=False)):
                    raise ValueError('Control no longer reproduces previous predictions exactly')
                item['original_predictions_sha256'] = sha256(original)
                item['original_prediction_parity'] = True
            else:
                binary = run/'proposal.fp32.tflite'
                meta = read_json(binary.with_suffix('.json'))
                if (meta['sha256']!=sha256(binary) or meta['parameters']!=2342
                        or meta['source_checkpoint_sha256']!=report['model_sha256']
                        or meta['manifest_sha256']!=sha256(root/'manifest.json')
                        or meta['decoded_parity_frames']!=len(x)
                        or not 0<=meta['max_tensor_delta']<=.002):
                    raise ValueError('Export identity or parity changed')
                item['export'] = dict(path=str(binary),metadata_sha256=sha256(binary.with_suffix('.json')),**meta)
                control = read_json(f'runs/balloon-dense-bg-sparse-seed{seed}-20261011/provenance.json')
                for key in ('seed','epochs','batch_size','learning_rate','parameters','script_sha256'):
                    if control[key]!=provenance[key]:
                        raise ValueError('Matched training configuration changed')
            trials[mode][str(seed)] = item
    panels = {}
    for panel in PANELS:
        path = Path(f'runs/balloon-dense-bg-scenes-{panel}-20261011.json')
        doc = read_json(path)
        previous_path = Path(f'runs/balloon-dense-scenes-{panel}-20261011.json')
        previous = read_json(previous_path)
        for key in ('classifier_sha256','queue_sha256','review_sha256','selection',
                    'split_max_aspect','proposal_threshold','classifier_threshold','matching_iou'):
            if doc[key]!=previous[key]:
                raise ValueError('Scene comparison configuration changed')
        if doc['classifier_sha256']!=sha256(CLASSIFIER):
            raise ValueError('Classifier changed')
        for seed,identity in doc['dense_models'].items():
            expected = trials['complete'][seed]['export']
            if any(identity[key]!=expected[key] for key in ('sha256','metadata_sha256')):
                raise ValueError('Wrong evaluated dense checkpoint')
        # Ignore nondeterministic host timing; compare every detection and truth.
        for a,b in zip(doc['results']['baseline']['frames'],previous['results']['baseline']['frames'],strict=True):
            check_baseline(a,b)
        panels[panel] = dict(path=str(path),sha256=sha256(path),previous_path=str(previous_path),
            previous_sha256=sha256(previous_path),frames=len(doc['results']['baseline']['frames']),
            current={name:checked_metrics(r) for name,r in doc['results'].items()},
            sparse={name:checked_metrics(r) for name,r in previous['results'].items()},
            added_overlap={name:added_overlap(doc['results']['baseline']['frames'],doc['results'][name]['frames'])
                           for name in ('dense47','dense48')})
    return dict(trials=trials,panels=panels,complete_review=new['complete_background'],
        complete_fit_frames=len(approved),unchanged_holdout_frames=int((folds==0).sum()),
        extra_heat_cells=int((cy[:,:,:,2:4]-y[:,:,:,2:4]).sum()),
        source_manifest_sha256=sha256(SPARSE/'manifest.json'),
        treatment_manifest_sha256=sha256(COMPLETE/'manifest.json'),script_sha256=sha256(__file__),
        decision='Do not promote either dense-search model. Background labels improve localization and some FP counts versus sparse dense training, but full-scene precision remains worse than the no-dense baseline.',
        next_direction='Test box-overlap-aware training loss on these fixed training labels; unchanged development thresholds. Do not select a new threshold from development failures.',
        caveats=['Assistant visual completeness review, not independent human annotation.',
                 'Complete-fit proposal metrics are training diagnostics, not validation.',
                 'Grouped holdout shares environments and related classifier sources.',
                 'Development panels overlap and have been repeatedly inspected; not independent qualification.',
                 'Host export parity does not establish Pi speed or camera-inclusive readiness.'],
        test_evaluated=False,deployment_approved=False,pi_tested=False,qualification_achieved=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    write_json(a.output,summarize())
