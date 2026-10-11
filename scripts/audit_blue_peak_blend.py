"""Recount host/Pi detections and counterbalanced timing, without flight claims."""
from pathlib import Path

import numpy as np

from dtr.data import read_json,sha256,write_json
from scripts.build_blue_peak_blend import OUTPUT as ROOT,ARMS,checked_scenes
from scripts.evaluate_blue_peak_blend import OUTPUT as SCENES
from scripts.evaluate_red_blue_development import LABELS,detection_counts,metrics
from scripts.pi_blue_peak_blend import ORDER,TIMINGS
from scripts.pi_current_balloon_pipeline import compare
from scripts.pi_red_blue_bench import statistics,verify_bundle

OUTPUT = Path('research/balloon-red-blue-20261008/blue-peak-blend-decision-20261011.json')


def audit():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    host,_ = checked_scenes()
    bundle = verify_bundle(ROOT)
    pi_path = ROOT/'pi-paired.json'
    pi = read_json(pi_path)
    if (pi['machine'] != 'armv6l' or pi['bundle_sha256'] != sha256(ROOT/'bundle.json')
            or pi['script_sha256'] != sha256('scripts/pi_blue_peak_blend.py')
            or pi['model_sha256'] != bundle['classifier_sha256']
            or pi['localizer_sha256'] != bundle['localizer_sha256']
            or pi['head_localizer_sha256'] != bundle['head_localizer_sha256']
            or pi['runtime_library_sha256'] != '54ed7fdda0b5b5f6db05a538eb800b7bc7ecd70d32f67dc13d1e78817d8b653a'
            or pi['region_library_sha256'] != 'd7620537d7e9850f0f794a26facc188063f289c6ad9f5d793693ad5715e43303'
            or tuple(b['policy'] for b in pi['blocks']) != ORDER
            or any(pi[k] is not False for k in ('camera_used','model_replaced','deployment_approved',
                'qualification_achieved','esp32_commands_sent','motor_commands_sent'))):
        raise ValueError('Pi identity, block order or scope changed')
    frames = read_json(ROOT/'decoder-frames.json')
    combined = {policy:[] for policy in ARMS}
    blocks = []
    for block in pi['blocks']:
        if block['frames'] != 141 or len(block['rows']) != 141:
            raise ValueError('Incomplete timing block')
        policy = block['policy']
        totals = {}
        for row,source in zip(block['rows'],frames,strict=True):
            if any(row[k] != source[k] for k in ('panel','id','source','frame_sha256','truth')):
                raise ValueError('Pi scene identity changed')
            expected = source['expected'][policy]
            if compare(row['detections'],expected) != (row['score_delta'],row['box_delta']):
                raise ValueError('Pi parity receipt changed')
            counts = detection_counts(row['truth'],row['detections'])
            if counts != detection_counts(row['truth'],expected):
                raise ValueError('Device numeric differences changed detection accuracy')
            if row['neural_calls'] != len(expected) or row['localizer_calls'] != 1:
                raise ValueError('Unexpected inference workload')
            if (not all(np.isfinite(row[k]) and row[k] >= 0 for k in TIMINGS)
                    or not np.isclose(row['processing_ms'],sum(row[k] for k in TIMINGS[1:]),atol=1e-5,rtol=0)):
                raise ValueError('Invalid stage timings')
            total = totals.setdefault(row['panel'],{c:dict(tp=0,fp=0,fn=0) for c in LABELS})
            for color in LABELS:
                for key,value in counts[color].items():
                    total[color][key] += value
        for panel,counts in totals.items():
            if metrics(counts) != host['panels'][panel][ARMS[policy]]['metrics']:
                raise ValueError('Pi scene totals do not reproduce host')
        timing = {k:statistics([r[k] for r in block['rows']]) for k in TIMINGS}
        if timing != block['timing']:
            raise ValueError('Timing summaries changed')
        combined[policy].extend(block['rows'])
        blocks.append(dict(policy=policy,timing=timing,health_before=block['health_before'],health_after=block['health_after']))
    policies = {}
    for policy,rows in combined.items():
        timing = {k:statistics([r[k] for r in rows]) for k in TIMINGS}
        policies[policy] = dict(timing=timing,processing_only_fps=1000/timing['processing_ms']['mean_ms'],
            timed_frames=len(rows),neural_calls=sum(r['neural_calls'] for r in rows),
            max_score_delta=max(r['score_delta'] for r in rows),max_box_delta=max(r['box_delta'] for r in rows))
    original = policies['original']['timing']['processing_ms']['mean_ms']
    for policy in ('blend','head_blend'):
        value = policies[policy]['timing']['processing_ms']['mean_ms']
        policies[policy]['processing_time_change_percent'] = 100*(value/original-1)
    healthy = all(b[k]['throttled'] == 'throttled=0x0' for b in blocks for k in ('health_before','health_after'))
    write_json(OUTPUT,dict(policies=policies,blocks=blocks,host_panels=host['panels'],
        no_throttle_flags=healthy,host_scene_sha256=sha256(SCENES),bundle_sha256=sha256(ROOT/'bundle.json'),
        pi_report=dict(path=str(pi_path),sha256=sha256(pi_path)),script_sha256=sha256(__file__),
        tests_log_sha256=sha256('runs/blue-peak-blend-final-tests-20261011.log'),
        decision='Retain decoder candidates as explicit research options only; no active default replacement or flight qualification.',
        scope='Sequential ABCCBA warmed ARMv6 processing-only replay.282timed rows per policy are repeats of141related development images, not282 independent scenes. Capture/display/serial and live motion excluded.',
        test_evaluated=False,live_candidate_tested=False,deployment_approved=False,qualification_achieved=False,
        esp32_commands_sent=False,motor_commands_sent=False))
    print(dict(policies=policies,healthy=healthy))


if __name__ == '__main__':
    audit()
