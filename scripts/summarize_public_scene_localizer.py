"""Recount scene decisions and verify paired TRAIN qualification gates from saved predictions."""
from pathlib import Path

from dtr.data import read_json, sha256, write_json
from scripts.audit_public_scene_localizer import OUTPUT as AUDIT
from scripts.summarize_public_balloon_trial import checked_metrics
from scripts.diagnose_clutter_stage_attribution import difference

OUTPUT = Path('research/balloon-red-blue-20261008/public-scene-localizer-decision-20261011.json')


def summarize():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    report = read_json(AUDIT)
    if any(sha256(path) != h for path, h in report['inputs_sha256'].items()):
        raise ValueError('Evaluation source changed')
    raw_path = report['raw']['path']
    if sha256(raw_path) != report['raw']['sha256']:
        raise ValueError('Predictions changed')
    for record in report['training'].values():
        for filename, h in record['artifacts_sha256'].items():
            if sha256(Path(record['run'])/filename) != h:
                raise ValueError('Evaluated checkpoint changed')
    panels = {}
    for panel, variants in read_json(raw_path).items():
        expected_frames = {'original': 64, 'window1': 6, 'window2': 12}[panel]
        panels[panel] = {}
        for name, result in variants.items():
            if len(result['frames']) != expected_frames or any(f['neural_calls'] > 12 for f in result['frames']):
                raise ValueError('Frame coverage or budget changed')
            measured = checked_metrics(result)
            effect = difference(result, variants['parent_current'])
            safe = not effect['lost_targets'] and all(effect['count_delta'][c]['fp'] <= 0 for c in measured)
            stored = report['panels'][panel][name]
            if measured != stored['metrics'] or effect != stored['effect'] or safe != stored['no_regression']:
                raise ValueError('Fresh counts or target-retention decision disagree')
            panels[panel][name] = dict(metrics=measured, no_regression=safe,
                lost_targets=effect['lost_targets'], gained_targets=effect['gained_targets'],
                false_positive_change={c: effect['count_delta'][c]['fp'] for c in measured})
            if name.startswith('scenes'):
                control = name.replace('scenes', 'control', 1)
                comparison = difference(result, variants[control])
                if comparison != stored['against_matched_control']:
                    raise ValueError('Matched control comparison does not reproduce')
                panels[panel][name]['against_matched_control'] = comparison
    gates = {policy: all(all(p[f'scenes{seed}_{policy}']['no_regression'] for p in panels.values()) and
        any(p[f'scenes{seed}_{policy}']['gained_targets'] for p in panels.values())
        for seed in (47, 48)) for policy in ('current', 'reserved')}
    if gates != report['eligible_for_separate_export_development']:
        raise ValueError('Paired eligibility gate changed')
    write_json(OUTPUT, dict(panels=panels, eligible_for_separate_export_development=gates,
        audit_sha256=sha256(AUDIT), raw_sha256=sha256(raw_path), script_sha256=sha256(__file__),
        decision=report['decision'],
        scope='Independent recount of stored82TRAINscene predictions, identity-level target retention and paired controls. No new model inference, threshold tuning, held-out accuracy or Pi claim.',
        original_holdout_evaluated=False, development_evaluated=False, test_evaluated=False,
        pi_tested=False, deployment_approved=False, qualification_achieved=False))
    print(dict(eligible=gates), flush=True)


if __name__ == '__main__':
    summarize()
