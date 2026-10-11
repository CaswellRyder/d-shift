"""Audit actual Pi predictions and timings, never infer live or flight readiness."""
from collections import Counter
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from scripts.evaluate_red_blue_development import LABELS, detection_counts, metrics
from scripts.pi_current_balloon_pipeline import compare
from scripts.pi_red_blue_bench import statistics, verify_bundle

ROOT = Path('runs/current-balloon-pipeline-pi-20261011')
BASE = Path('research/balloon-red-blue-20261008')


def audit():
    output = BASE/'current-balloon-pipeline-pi-20261011.json'
    if output.exists():
        raise FileExistsError(output)
    bundle = verify_bundle(ROOT)
    path = ROOT/'pi-replay.json'
    pi = read_json(path)
    if (pi['machine'] != 'armv6l' or pi['frames'] != 141 or len(pi['rows']) != 141
            or pi['bundle_sha256'] != sha256(ROOT/'bundle.json')
            or pi['script_sha256'] != sha256('scripts/pi_current_balloon_pipeline.py')
            or pi['model_sha256'] != bundle['classifier_sha256']
            or pi['localizer_sha256'] != bundle['localizer_sha256']
            or pi['runtime_library_sha256'] != '54ed7fdda0b5b5f6db05a538eb800b7bc7ecd70d32f67dc13d1e78817d8b653a'
            or pi['region_library_sha256'] != 'd7620537d7e9850f0f794a26facc188063f289c6ad9f5d793693ad5715e43303'
            or any(pi[k] is not False for k in ('camera_used', 'deployment_approved', 'esp32_commands_sent', 'motor_commands_sent'))
            or any(sha256(p) != h for p, h in bundle['source_reports'].items())):
        raise ValueError('Unexpected device, model, runtime or evidence scope')
    counts, max_scores, max_boxes = {}, 0., 0.
    for row, source in zip(pi['rows'], read_json(ROOT/'frames.json'), strict=True):
        if any(row[k] != source[k] for k in ('panel', 'id', 'source', 'frame_sha256', 'truth')):
            raise ValueError('Device frame identities differ')
        score_delta, box_delta = compare(row['detections'], source['expected'])
        if row['score_delta'] != score_delta or row['box_delta'] != box_delta:
            raise ValueError('Recorded numerical parity does not reproduce')
        max_scores, max_boxes = max(max_scores, score_delta), max(max_boxes, box_delta)
        if row['neural_calls'] != len(row['detections']) or row['localizer_calls'] != 1:
            raise ValueError('Inference workload mismatch')
        current = detection_counts(row['truth'], row['detections'])
        if current != detection_counts(row['truth'], source['expected']):
            raise ValueError('Pi numerical differences changed object metrics')
        totals = counts.setdefault(row['panel'], {c: Counter(tp=0, fp=0, fn=0) for c in LABELS})
        for color in LABELS:
            totals[color].update(current[color])
    timing = {key: statistics([r[key] for r in pi['rows']]) for key in pi['timing']}
    fps = 1000/np.mean([r['processing_ms'] for r in pi['rows']])
    if (timing != pi['timing'] or fps != pi['processing_only_fps']
            or max_scores != pi['max_score_delta'] or max_boxes != pi['max_box_delta']):
        raise ValueError('Recorded timing/parity summaries do not reproduce')
    geometry = {}
    for stage in ('training', 'development'):
        p = BASE/f'blue-component-box-{stage}-20261011.json'
        doc = read_json(p)
        if sha256(doc['raw']['path']) != doc['raw']['sha256']:
            raise ValueError('Geometry trial changed')
        geometry[stage] = dict(receipt_sha256=sha256(p),
            changed_boxes=sum(v['component']['changed_boxes'] for v in doc['panels'].values()),
            fallback_reasons=dict(Counter(d['box_refinement_reason'] for panel in read_json(doc['raw']['path']).values()
                for row in panel['component']['frames'] for d in row['detections'] if 'box_refinement_reason' in d)))
    result = dict(pi_report=dict(path=str(path), sha256=sha256(path)), bundle_sha256=sha256(ROOT/'bundle.json'),
        models=dict(classifier_sha256=pi['model_sha256'], localizer_sha256=pi['localizer_sha256']),
        frames=141, panel_metrics={p: metrics(c) for p, c in counts.items()}, timing=timing,
        processing_only_fps=float(fps), max_score_delta=max_scores, max_box_delta=max_boxes,
        classifier_calls=sum(r['neural_calls'] for r in pi['rows']), localizer_calls=141,
        health_before=pi['health_before'], health_after=pi['health_after'], geometry_trial=geometry,
        decision='Full research pipeline executes on original ARMv6 Pi and reproduces development decisions/metrics. Geometry correction was inactive on every tested frame; do not promote it. No live-camera qualification or94-percent claim; active deployment unchanged.',
        tests_log_sha256=sha256('runs/current-balloon-pipeline-tests-20261011.log'),
        script_sha256=sha256(__file__), helpers_sha256={p: sha256(p) for p in
            ('scripts/pi_current_balloon_pipeline.py', 'scripts/evaluate_red_blue_development.py',
             'scripts/pi_red_blue_bench.py')},
        scope='Warmed single-pipeline replay, camera capture/display/serial excluded. Not a matched comparison against deployed pipeline.',
        deployment_approved=False, qualification_achieved=False, test_evaluated=False,
        live_candidate_tested=False, esp32_commands_sent=False, motor_commands_sent=False)
    write_json(output, result)
    print(dict(fps=fps, timing=timing, geometry_trial=geometry, panel_metrics=result['panel_metrics']))


if __name__ == '__main__':
    audit()
