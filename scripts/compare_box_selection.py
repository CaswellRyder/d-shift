"""Post-hoc diagnostic: correct geometry without resurrecting parent-rejected boxes.

Not a deployment policy. Original winners can still move onto the same object;
this evaluates whether refinement gains survive fixed instance selection.
"""
import argparse
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from scripts.evaluate_red_blue_development import LABELS, detection_counts, metrics


def locked_selection(parent, refined):
    if len(parent) != len(refined):
        raise ValueError('Proposal count mismatch')
    rows = []
    for old, new in zip(parent, refined):
        box = new['box']
        if (old['box'] != new['proposal_box'] or old['crop_box'] != new['crop_box']
                or old['label'] != new['label'] or old['scores'] != new['scores']
                or old['raw_accepted'] != new['raw_accepted']
                or len(box) != 4 or not np.isfinite(box).all()
                or not 0 <= box[0] < box[2] <= 320 or not 0 <= box[1] < box[3] <= 240):
            raise ValueError('Proposal identity, class predictions or corrected geometry changed')
        rows.append(dict(old,box=box,proposal_box=old['box'],
            box_correction_applied=new['box_correction_applied'], selection='locked_parent'))
    return rows


def compare(report):
    if (report.get('test_evaluated') is not False or report.get('deployment_approved') is not False
            or report.get('max_class_probability_delta',float('inf')) > 1e-5):
        raise ValueError('Require frozen-class development-only box report')
    parent, refined = report['results']['parent']['frames'],report['results']['refined']['frames']
    if len(parent) != len(refined) or not parent:
        raise ValueError('Frame count mismatch')
    total = {label:dict(tp=0,fp=0,fn=0) for label in LABELS}
    frames = []
    for old,new in zip(parent,refined):
        if old['source'] != new['source'] or old['truth'] != new['truth']:
            raise ValueError('Frame identity or truth mismatch')
        if detection_counts(old['truth'],old['detections']) != old['counts']:
            raise ValueError('Parent counts do not reproduce')
        found = locked_selection(old['detections'],new['detections'])
        counts = detection_counts(old['truth'],found)
        for label in LABELS:
            for key in ('tp','fp','fn'):
                total[label][key] += counts[label][key]
        frames.append(dict(source=old['source'],truth=old['truth'],detections=found,counts=counts))
    return dict(metrics=metrics(total),frames=frames,
        parent_metrics=report['results']['parent']['metrics'],
        refined_selection_metrics=report['results']['refined']['metrics'],
        model_sha256=report['model_sha256'],panel=report['panel'],script_sha256=sha256(__file__),
        matcher_sha256=sha256('scripts/evaluate_red_blue_development.py'),
        scope='Post-hoc development diagnostic: corrected boxes, identical parent winners; no new inference',
        test_evaluated=False,deployment_approved=False,pi_timing_measured=False)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    result=compare(read_json(args.report))
    result['source_report_sha256']=sha256(args.report)
    write_json(args.output,result)
    print(result['metrics'])
