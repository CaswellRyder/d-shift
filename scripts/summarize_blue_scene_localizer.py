"""Independently recount matched-scene results and retain identity-level regressions."""
from pathlib import Path

from dtr.data import read_json, sha256, write_json
from scripts.audit_indoor5_outcomes import compare_frames
from scripts.blue_scene_localizer_data import PROTOCOL
from scripts.evaluate_blue_scene_localizer import nonregression
from scripts.summarize_balloon_mask_convergence import meets_target
from scripts.summarize_public_balloon_trial import checked_metrics

DIRECTORY = Path('research/balloon-red-blue-20261008')


def decision(panels):
    result = {}
    for seed in (47, 48):
        key, control = f'background{seed}', f'control{seed}'
        result[key] = dict(
            nonregression_vs_parent=all(p[key]['nonregression'] for p in panels.values()),
            retains_parent_target_identities=all(not p[key]['lost_parent_targets'] for p in panels.values()),
            nonregression_vs_control=all(nonregression(p[control]['metrics'], p[key]['metrics']) for p in panels.values()),
            meets_94_every_panel=all(meets_target(p[key]['metrics']) for p in panels.values()))
    return result


def summarize():
    output = DIRECTORY/'blue-scene-localizer-decision-20261011.json'
    if output.exists():
        raise FileExistsError(output)
    development_path = DIRECTORY/'blue-scene-localizer-development-20261011.json'
    training_path = DIRECTORY/'blue-scene-localizer-training-20261011.json'
    doc = read_json(development_path)
    raw_path = Path(doc['raw']['path'])
    if (sha256(raw_path) != doc['raw']['sha256'] or sha256(training_path) != doc['training_sha256']
            or doc['script_sha256'] != sha256('scripts/evaluate_blue_scene_localizer.py')
            or doc['protocol_sha256'] != sha256(PROTOCOL)
            or any(sha256(p) != h for p, h in doc['helpers_sha256'].items())):
        raise ValueError('Evaluation inputs or code changed')
    for model in doc['models'].values():
        path = Path(model['path'])
        if sha256(path) != model['sha256'] or sha256(path.with_suffix('.json')) != model['metadata_sha256']:
            raise ValueError('Evaluated model changed')
    panels = {}
    for panel, variants in read_json(raw_path).items():
        panels[panel] = {}
        for name, variant in variants.items():
            values = checked_metrics(variant)
            fresh = dict(metrics=values, **compare_frames(variant['frames'], variants['parent']['frames']),
                neural_calls=sum(r['neural_calls'] for r in variant['frames']), meets_target=meets_target(values),
                nonregression=nonregression(checked_metrics(variants['parent']), values))
            if fresh != doc['panels'][panel]['results'][name]:
                raise ValueError('Metrics or target identities fail independent recount')
            panels[panel][name] = fresh
    gates = decision(panels)
    result = dict(gates=gates,
        decision='Retain background47 as research candidate; do not replace Pi model. Background supervision reduced false detections on the67-frame panel, but benefit was not nonregressing across both seeds. No candidate reaches94 percent precision and recall on every development panel.',
        next_step='Address remaining blue false-positive sources and suppression/localization losses with TRAIN-only annotations. Keep current classifier and incumbent controls; do not tune against reserved qualification data.',
        development_sha256=sha256(development_path), training_sha256=sha256(training_path),
        raw_sha256=sha256(raw_path), script_sha256=sha256(__file__),
        tests=dict(passed=1314, skipped=2, warnings=6,
            log_sha256=sha256('runs/blue-scene-localizer-tests-20261011.log')),
        limitations=['All three panels are repeatedly inspected related development scenes, not independent qualification.',
            'Eight new complete-negative scenes are training evidence only.',
            'Unchanged2342-parameter architecture and14420-byte exports do not establish Pi latency.',
            'No Pi replacement, live candidate execution, ESP32 or motor commands.'],
        test_evaluated=False, pi_tested=False, deployment_approved=False, qualification_achieved=False)
    write_json(output, result)
    print(gates)


if __name__ == '__main__':
    summarize()
