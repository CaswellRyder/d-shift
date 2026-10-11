"""Verify matched L1 versus L1+GIoU training and unchanged full-scene tests."""
import argparse
from pathlib import Path

import numpy as np

from dtr.data import read_json,sha256,write_json
from scripts.audit_balloon_dense_proposals import audit
from scripts.summarize_dense_background_trial import check_baseline,COMPLETE
from scripts.summarize_balloon_fragments import PANELS
from scripts.summarize_public_balloon_trial import checked_metrics
from scripts.summarize_balloon_dense_scenes import added_overlap


def matched(control,treatment):
    for key in ('manifest_sha256','script_sha256','preparation_script_sha256','seed',
                'epochs','batch_size','learning_rate','parameters','fixed_final_epoch',
                'folds','groups','complete_background','loss_helper_sha256'):
        if control[key]!=treatment[key]:
            raise ValueError('Matched experiment differs outside box objective: '+key)
    if (control['box_objective']!='l1' or control['giou_weight']!=0
            or treatment['box_objective']!='l1_giou' or treatment['giou_weight']!=2):
        raise ValueError('Box objective or fixed weight changed')


def summarize():
    trials = {}
    for seed in (47,48):
        pair = {}
        for objective in ('l1','l1_giou'):
            run = Path(f'runs/balloon-dense-overlap-{objective}-seed{seed}-20261011')
            coverage = audit(COMPLETE,run)
            provenance = read_json(run/'provenance.json')
            report = read_json(run/'report.json')
            pair[objective] = dict(run=str(run),provenance=provenance,report=report,
                coverage=coverage['totals'],provenance_sha256=sha256(run/'provenance.json'),
                report_sha256=sha256(run/'report.json'))
        matched(pair['l1']['provenance'],pair['l1_giou']['provenance'])
        previous = Path(f'runs/balloon-dense-bg-complete-seed{seed}-20261011')
        if not np.array_equal(np.load(previous/'predictions.npy',allow_pickle=False),
                np.load(Path(pair['l1']['run'])/'predictions.npy',allow_pickle=False)):
            raise ValueError('Control does not reproduce earlier training exactly')
        pair['l1']['previous_predictions_sha256'] = sha256(previous/'predictions.npy')
        pair['l1']['previous_predictions_exact_match'] = True
        binary = Path(pair['l1_giou']['run'])/'proposal.fp32.tflite'
        meta = read_json(binary.with_suffix('.json'))
        if (meta['sha256']!=sha256(binary) or meta['parameters']!=2342
                or meta['bytes']!=14420 or meta['decoded_parity_frames']!=108
                or meta['source_checkpoint_sha256']!=pair['l1_giou']['report']['model_sha256']
                or meta['manifest_sha256']!=sha256(COMPLETE/'manifest.json')
                or not 0<=meta['max_tensor_delta']<=.002):
            raise ValueError('Export architecture or parity changed')
        pair['l1_giou']['export'] = dict(path=str(binary),metadata_sha256=sha256(binary.with_suffix('.json')),**meta)
        trials[str(seed)] = pair
    panels = {}
    for panel in PANELS:
        path = Path(f'runs/balloon-dense-overlap-scenes-{panel}-20261011.json')
        previous_path = Path(f'runs/balloon-dense-bg-scenes-{panel}-20261011.json')
        doc,previous = read_json(path),read_json(previous_path)
        for key in ('classifier_sha256','queue_sha256','review_sha256','selection',
                    'split_max_aspect','proposal_threshold','classifier_threshold','matching_iou',
                    'helpers_sha256','script_sha256'):
            if doc[key]!=previous[key]:
                raise ValueError('Evaluation changed outside dense weights: '+key)
        for a,b in zip(doc['results']['baseline']['frames'],previous['results']['baseline']['frames'],strict=True):
            check_baseline(a,b)
        for seed,identity in doc['dense_models'].items():
            expected = trials[seed]['l1_giou']['export']
            if any(identity[k]!=expected[k] for k in ('sha256','metadata_sha256')):
                raise ValueError('Scene checkpoint identity changed')
        panels[panel] = dict(path=str(path),sha256=sha256(path),previous_path=str(previous_path),
            previous_sha256=sha256(previous_path),frames=len(doc['results']['baseline']['frames']),
            l1={k:checked_metrics(r) for k,r in previous['results'].items()},
            l1_giou={k:checked_metrics(r) for k,r in doc['results'].items()},
            added_overlap={name:added_overlap(doc['results']['baseline']['frames'],doc['results'][name]['frames'])
                           for name in ('dense47','dense48')})
    return dict(trials=trials,panels=panels,script_sha256=sha256(__file__),
        training_loss_reference='https://arxiv.org/abs/1902.09630',
        loss_definition='Unchanged focal + 5*normalized-coordinate L1 + 2*mean(1-GIoU), reviewed centers only; no inference graph change.',
        decision='Do not promote either overlap-loss model. Both still lower blue precision versus the no-dense baseline on all three development panels; changes versus matched L1 are mixed.',
        next_direction='Audit additional eligible public TRAIN frames and object/background coverage before further model tuning. Existing full-scene false positives remain the limiting issue.',
        caveats=['Different total training losses are not directly comparable across objectives.',
                 'Grouped training holdout is not independent competition qualification.',
                 'Development panels overlap and are repeatedly inspected; report separately.',
                 'No Pi timing or camera-inclusive verification performed for these exports.'],
        test_evaluated=False,deployment_approved=False,pi_tested=False,qualification_achieved=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    write_json(a.output,summarize())
