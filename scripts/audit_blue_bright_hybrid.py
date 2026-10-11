"""Independently recount paired accuracy and ARMv6 timing; no deployment approval."""
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from scripts.audit_indoor5_outcomes import compare_frames
from scripts.evaluate_red_blue_development import LABELS, detection_counts, metrics
from scripts.pi_current_balloon_pipeline import compare
from scripts.pi_red_blue_bench import statistics, verify_bundle
from scripts.summarize_public_balloon_trial import checked_metrics

BASE = Path('research/balloon-red-blue-20261008')
ROOT = Path('runs/blue-bright-hybrid-pi-20261011')


def audit():
    output = BASE/'blue-bright-hybrid-decision-20261011.json'
    if output.exists():
        raise FileExistsError(output)
    source_metrics, receipts, retention = {}, {}, {}
    for stage in ('training', 'development'):
        path = BASE/f'blue-bright-hybrid-{stage}-20261011.json'
        doc = read_json(path)
        raw = Path(doc['raw']['path'])
        if (sha256(raw) != doc['raw']['sha256']
                or doc['script_sha256'] != sha256('scripts/evaluate_blue_bright_hybrid.py')
                or any(sha256(p) != h for p, h in doc['helpers_sha256'].items())
                or any(sha256(p) != h for p, h in doc['inputs'].items())):
            raise ValueError('Host comparison identity changed')
        retention[stage], source_metrics[stage] = {}, {}
        for panel, variants in read_json(raw).items():
            source_metrics[stage][panel] = {}
            for arm, value in variants.items():
                recalculated = checked_metrics(value)
                if recalculated != doc['panels'][panel][arm]['metrics']:
                    raise ValueError('Host metrics do not reproduce')
                source_metrics[stage][panel][arm] = recalculated
            change = compare_frames(variants['blue_bright']['frames'], variants['full']['frames'])
            if any(change[k] != doc['panels'][panel]['blue_bright'][k] for k in change):
                raise ValueError('Target retention does not reproduce')
            retention[stage][panel] = dict(lost=change['lost_parent_targets'], gained=change['gained_targets'])
        receipts[stage] = dict(path=str(path), sha256=sha256(path), raw_sha256=sha256(raw))
    bundle = verify_bundle(ROOT)
    pi_path = ROOT/'pi-paired.json'
    pi = read_json(pi_path)
    if (pi['machine'] != 'armv6l' or pi['bundle_sha256'] != sha256(ROOT/'bundle.json')
            or pi['script_sha256'] != sha256('scripts/pi_blue_bright_hybrid.py')
            or pi['model_sha256'] != bundle['classifier_sha256'] or pi['localizer_sha256'] != bundle['localizer_sha256']
            or pi['runtime_library_sha256'] != '54ed7fdda0b5b5f6db05a538eb800b7bc7ecd70d32f67dc13d1e78817d8b653a'
            or pi['region_library_sha256'] != 'd7620537d7e9850f0f794a26facc188063f289c6ad9f5d793693ad5715e43303'
            or [b['policy'] for b in pi['blocks']] != ['original', 'blue_bright', 'blue_bright', 'original']
            or any(pi[k] is not False for k in ('camera_used', 'model_replaced', 'deployment_approved',
                'qualification_achieved', 'esp32_commands_sent', 'motor_commands_sent'))):
        raise ValueError('Pi identity, block order or scope changed')
    reference = read_json(ROOT/'paired-frames.json')
    combined = dict(original=[], blue_bright=[])
    blocks = []
    for block in pi['blocks']:
        if block['frames'] != 141 or len(block['rows']) != 141:
            raise ValueError('Incomplete paired block')
        totals = {}
        for row, source in zip(block['rows'], reference, strict=True):
            if any(row[k] != source[k] for k in ('panel', 'id', 'source', 'frame_sha256', 'truth')):
                raise ValueError('Pi source identity changed')
            expected = source['expected'][block['policy']]
            delta = compare(row['detections'], expected)
            if delta != (row['score_delta'], row['box_delta']):
                raise ValueError('Pi parity report changed')
            counts = detection_counts(row['truth'], row['detections'])
            if counts != detection_counts(row['truth'], expected):
                raise ValueError('Pi numeric differences changed accuracy')
            if row['neural_calls'] != len(expected) or row['localizer_calls'] != 1:
                raise ValueError('Unexpected inference workload')
            total = totals.setdefault(row['panel'], {c: dict(tp=0, fp=0, fn=0) for c in LABELS})
            for color in LABELS:
                for key, value in counts[color].items():
                    total[color][key] += value
        arm = 'full' if block['policy'] == 'original' else 'blue_bright'
        for panel, counts in totals.items():
            if metrics(counts) != source_metrics['development'][panel][arm]:
                raise ValueError('Pi aggregate accuracy changed')
        timing = {key: statistics([r[key] for r in block['rows']]) for key in block['timing']}
        if timing != block['timing']:
            raise ValueError('Pi timings do not reproduce')
        combined[block['policy']].extend(block['rows'])
        blocks.append(dict(policy=block['policy'], timing=timing,
            health_before=block['health_before'], health_after=block['health_after']))
    policies = {}
    for policy, rows in combined.items():
        timing = {key: statistics([r[key] for r in rows]) for key in pi['blocks'][0]['timing']}
        policies[policy] = dict(timing=timing, processing_only_fps=1000/timing['processing_ms']['mean_ms'],
            timed_frames=len(rows), neural_calls=sum(r['neural_calls'] for r in rows),
            max_score_delta=max(r['score_delta'] for r in rows), max_box_delta=max(r['box_delta'] for r in rows))
    old = policies['original']['timing']['processing_ms']['mean_ms']
    new = policies['blue_bright']['timing']['processing_ms']['mean_ms']
    identical = all(v['full'] == v['blue_bright'] for stage in source_metrics.values() for v in stage.values())
    retained = all(not p['lost'] for panels in retention.values() for p in panels.values())
    healthy = all(b[h]['throttled'] == 'throttled=0x0' for b in blocks for h in ('health_before', 'health_after'))
    if not np.isfinite(old+new) or min(old, new) <= 0:
        raise ValueError('Invalid speed evidence')
    result = dict(policies=policies, blocks=blocks, host_metrics=source_metrics, target_retention=retention,
        identical_accuracy_counts=identical, retained_every_previous_target=retained, no_throttle_flags=healthy,
        processing_time_reduction_percent=100*(1-new/old), processing_fps_ratio=old/new,
        receipts=receipts, pi_report=dict(path=str(pi_path), sha256=sha256(pi_path)),
        bundle_sha256=sha256(ROOT/'bundle.json'), script_sha256=sha256(__file__),
        decision='Retain as opt-in research candidate only. Use measured speed and all-scene target retention together; repeated development accuracy is not independent qualification. Do not replace deployed defaults.',
        tests_log_sha256=sha256('runs/blue-bright-hybrid-final-tests-20261011.log'),
        scope='ABBA sequential processing-only Pi replay. Two repetitions per policy are timing repeats, not282 independent scenes. Camera, display, serial and flight behavior excluded.',
        test_evaluated=False, deployment_approved=False, qualification_achieved=False, live_candidate_tested=False,
        esp32_commands_sent=False, motor_commands_sent=False)
    write_json(output, result)
    print(dict(policies=policies, time_reduction=result['processing_time_reduction_percent'],
        identical_counts=identical, retained_targets=retained, healthy=healthy))


if __name__ == '__main__':
    audit()
