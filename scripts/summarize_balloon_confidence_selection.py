"""Recount selection evidence and separate Pi microtiming from live readiness."""
import math
from pathlib import Path

from dtr.data import read_json, sha256, write_json
from scripts.audit_indoor5_outcomes import compare_frames
from scripts.evaluate_blue_scene_localizer import nonregression
from scripts.summarize_public_balloon_trial import checked_metrics

BASE = Path('research/balloon-red-blue-20261008')
ROOT = Path('runs/balloon-confidence-selection-pi-20261011')


def summarize():
    output = BASE/'balloon-confidence-selection-decision-20261011.json'
    if output.exists():
        raise FileExistsError(output)
    stages, receipts = {}, {}
    for stage in ('training', 'development'):
        path = BASE/f'balloon-confidence-selection-{stage}-20261011.json'
        doc = read_json(path)
        raw = Path(doc['raw']['path'])
        if (sha256(raw) != doc['raw']['sha256']
                or doc['script_sha256'] != sha256('scripts/evaluate_balloon_confidence_selection.py')
                or any(sha256(p) != h for p, h in doc['helpers_sha256'].items())):
            raise ValueError('Selection evidence or code changed')
        stages[stage] = {}
        for panel, variants in read_json(raw).items():
            stages[stage][panel] = {}
            for name, variant in variants.items():
                parent = variants[name.split('/')[0]+'/current']
                values = checked_metrics(variant)
                comparison = compare_frames(variant['frames'], parent['frames'])
                stored = doc['panels'][panel]['results'][name]
                if (values != stored['metrics'] or any(stored[k] != v for k, v in comparison.items())
                        or stored['nonregression'] != nonregression(checked_metrics(parent), values)
                        or stored['neural_calls'] != sum(f['neural_calls'] for f in variant['frames'])):
                    raise ValueError('Counts, identity-level changes or workload do not reproduce')
                stages[stage][panel][name] = dict(metrics=values,
                    retains_previous_targets=not comparison['lost_parent_targets'], nonregression=stored['nonregression'])
        receipts[stage] = dict(path=str(path), sha256=sha256(path), raw_sha256=sha256(raw))
    pi_path = ROOT/'pi-selection.json'
    pi = read_json(pi_path)
    bundle = read_json(ROOT/'bundle.json')
    if (pi['machine'] != 'armv6l' or pi['parity_rows'] != 282
            or pi['bundle_sha256'] != sha256(ROOT/'bundle.json')
            or pi['inputs_sha256'] != sha256(ROOT/'inputs.json')
            or pi['script_sha256'] != sha256('scripts/pi_balloon_confidence_selection_bench.py')
            or [b['policy'] for b in pi['blocks']] != ['current', 'confidence', 'confidence', 'current']
            or any(b['calls'] != 4230 for b in pi['blocks'])
            or any(sha256(p) != h for p, h in bundle['source_sha256'].items())
            or any(sha256(ROOT/p) != h for p, h in bundle['files'].items())
            or any(pi[k] is not False for k in ('camera_used', 'neural_inference_used', 'model_replaced',
                'esp32_commands_sent', 'motor_commands_sent'))):
        raise ValueError('Pi benchmark identity or scope changed')
    timing = {}
    for key in ('current', 'confidence'):
        blocks = [b for b in pi['blocks'] if b['policy'] == key]
        if not all(math.isfinite(b['timing']['mean_ms']) and b['timing']['mean_ms'] > 0 for b in blocks):
            raise ValueError('Invalid timings')
        timing[key] = dict(mean_ms=sum(b['timing']['mean_ms']*b['calls'] for b in blocks)/8460,
            calls=8460, block_means_ms=[b['timing']['mean_ms'] for b in blocks])
    gates = {key: all(values[key]['nonregression'] and values[key]['retains_previous_targets']
        for panels in stages.values() for values in panels.values())
        for key in ('parent/confidence', 'background47/confidence')}
    result = dict(stages=stages, receipts=receipts, nonregression_all_screens=gates,
        pi=dict(path=str(pi_path), sha256=sha256(pi_path), timing=timing,
            health_before=pi['health_before'], health_after=pi['health_after'], parity_rows=282),
        decision='Retain parent-localizer confidence selection as research candidate. It preserves every previous target on the64 TRAIN scenes and all three development panels, recovers one blue target on panel2 and removes one blue false detection on panel3. Do not deploy the background47 combination: one extra false detection versus its own control on panel3. No94-percent qualification or full-pipeline speed claim.',
        tests_log_sha256=sha256('runs/balloon-confidence-selection-final-tests-20261011.log'),
        script_sha256=sha256(__file__),
        scope='Selection-only post-hoc development research and separately measured ARMv6 microtiming. Real independent scenes and camera-inclusive tests still needed.',
        deployment_approved=False, qualification_achieved=False, live_candidate_tested=False,
        test_evaluated=False, pi_model_replaced=False, esp32_commands_sent=False, motor_commands_sent=False)
    write_json(output, result)
    print(dict(gates=gates, pi_timing=timing))


if __name__ == '__main__':
    summarize()
