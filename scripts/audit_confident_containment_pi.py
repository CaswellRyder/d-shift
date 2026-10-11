"""Audit isolated ARMv6 parity, per-panel counts, workload and ABBA timings."""
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from scripts.build_confident_containment_pi import OUTPUT as ROOT
from scripts.evaluate_current_confident_containment import OUTPUT as HOST
from scripts.evaluate_red_blue_development import LABELS, detection_counts, metrics
from scripts.pi_blue_peak_blend import TIMINGS
from scripts.pi_confident_containment import ORDER, compare_all
from scripts.pi_red_blue_bench import statistics, verify_bundle

OUTPUT = Path('research/balloon-red-blue-20261008/current-containment-pi-decision-20261011.json')


def validate_row(row, source, policy):
    if any(row[k] != source[k] for k in ('panel', 'id', 'source', 'frame_sha256', 'truth')):
        raise ValueError('Replay scene identity changed')
    expected = source['expected'][policy]
    if compare_all(row['detections'], expected) != (row['score_delta'], row['box_delta']):
        raise ValueError('Stored parity deltas changed')
    if row['neural_calls'] != len(expected) or row['localizer_calls'] != 1:
        raise ValueError('Replay inference workload changed')
    if (not all(np.isfinite(row[k]) and row[k] >= 0 for k in (*TIMINGS, 'containment_extra_ms'))
            or row['containment_extra_ms'] > row['selection_ms']
            or not np.isclose(row['processing_ms'], sum(row[k] for k in TIMINGS[1:]), atol=1e-5, rtol=0)):
        raise ValueError('Invalid stage timings')
    actual = detection_counts(row['truth'], row['detections'])
    if actual != detection_counts(source['truth'], expected):
        raise ValueError('Device numeric differences changed accuracy')
    return actual


def audit():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    bundle, host = verify_bundle(ROOT), read_json(HOST)
    path = ROOT/'pi-paired.json'
    report = read_json(path)
    if (bundle['host_scenes_sha256'] != sha256(HOST) or not host['eligible_for_pi_comparison']
            or report['bundle_sha256'] != sha256(ROOT/'bundle.json') or report['machine'] != 'armv6l'
            or report['script_sha256'] != sha256('scripts/pi_confident_containment.py')
            or report['classifier_sha256'] != bundle['classifier_sha256']
            or report['localizer_sha256'] != bundle['head_localizer_sha256']
            or report['runtime_library_sha256'] != '54ed7fdda0b5b5f6db05a538eb800b7bc7ecd70d32f67dc13d1e78817d8b653a'
            or report['region_library_sha256'] != 'd7620537d7e9850f0f794a26facc188063f289c6ad9f5d793693ad5715e43303'
            or tuple(b['policy'] for b in report['blocks']) != ORDER
            or any(report[k] is not False for k in ('camera_used', 'model_replaced', 'esp32_commands_sent',
                'motor_commands_sent', 'deployment_approved', 'qualification_achieved'))):
        raise ValueError('Pi report scope/identity changed')
    frames = read_json(ROOT/'containment-frames.json')
    if len(frames) != 205 or any(sha256(ROOT/f['path']) != f['replay_file_sha256'] for f in frames):
        raise ValueError('Replay images changed')
    combined, blocks = {p: [] for p in ORDER}, []
    for block in report['blocks']:
        if len(block['rows']) != 205 or block['frames'] != 205:
            raise ValueError('Incomplete ABBA block')
        totals = {}
        for row, frame in zip(block['rows'], frames, strict=True):
            counted = validate_row(row, frame, block['policy'])
            total = totals.setdefault(row['panel'], {c: dict(tp=0, fp=0, fn=0) for c in LABELS})
            for c in LABELS:
                for key in ('tp', 'fp', 'fn'):
                    total[c][key] += counted[c][key]
        for panel, total in totals.items():
            expected = host['panels'][panel]['before' if block['policy']=='parent' else 'after']
            if metrics(total) != expected:
                raise ValueError('Pi panel metrics changed')
        timing = {k: statistics([r[k] for r in block['rows']]) for k in (*TIMINGS, 'containment_extra_ms')}
        if timing != block['timing']:
            raise ValueError('Reported timing summaries changed')
        combined[block['policy']].extend(block['rows'])
        blocks.append(dict(policy=block['policy'], timing=timing,
            health_before=block['health_before'], health_after=block['health_after']))
    policies = {}
    for policy, rows in combined.items():
        timing = {k: statistics([r[k] for r in rows]) for k in (*TIMINGS, 'containment_extra_ms')}
        policies[policy] = dict(timing=timing, timed_frames=len(rows),
            processing_only_fps=1000/timing['processing_ms']['mean_ms'],
            neural_calls=sum(r['neural_calls'] for r in rows),
            localizer_calls=sum(r['localizer_calls'] for r in rows),
            max_score_delta=max(r['score_delta'] for r in rows), max_box_delta=max(r['box_delta'] for r in rows))
    if policies['parent']['neural_calls'] != policies['confidence']['neural_calls']:
        raise ValueError('Unmatched policy workload')
    change = 100*(policies['confidence']['timing']['processing_ms']['mean_ms'] /
        policies['parent']['timing']['processing_ms']['mean_ms']-1)
    healthy = all(b[k]['throttled'] == 'throttled=0x0' for b in blocks for k in ('health_before', 'health_after'))
    result = dict(policies=policies, blocks=blocks, processing_time_change_percent=change,
        no_throttle_flags=healthy, host_scene_sha256=sha256(HOST), bundle_sha256=sha256(ROOT/'bundle.json'),
        pi_report=dict(path=str(path), sha256=sha256(path)), script_sha256=sha256(__file__),
        decision='Retain confidence containment as an opt-in research candidate. TRAIN suppression improvements reproduce on Pi; development accuracy unchanged. No active deployment change.',
        scope='ABBA processing-only replay:205scenes per block,410timed rows per policy. Same images repeated; not independent scene evidence. Excludes capture/display/serial and live target tracking. Different workload from earlier141scene benchmarks.',
        test_evaluated=False, live_camera_tested=False, deployment_approved=False, qualification_achieved=False,
        esp32_commands_sent=False, motor_commands_sent=False)
    write_json(OUTPUT, result)
    print(dict(policies=policies, processing_time_change_percent=change, no_throttle_flags=healthy), flush=True)


if __name__ == '__main__':
    audit()
