"""Recount fixed final-checkpoint scene outcomes and reject unsupported promotion."""
from pathlib import Path

from dtr.data import read_json, sha256, write_json
from scripts.audit_indoor5_outcomes import compare_frames
from scripts.evaluate_blue_scene_localizer import nonregression
from scripts.evaluate_runtime_balanced_trial import BASE, OUTPUT as DEVELOPMENT, RAW
from scripts.export_runtime_balanced_trial import checked_trial, TRIAL
from scripts.summarize_balloon_mask_convergence import meets_target
from scripts.summarize_public_balloon_trial import checked_metrics

OUTPUT = BASE/'runtime-balanced-decision-20261011.json'


def summarize():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    checked_trial()
    report = read_json(DEVELOPMENT)
    if (report['raw']['sha256'] != sha256(RAW) or report['training_trial_sha256'] != sha256(TRIAL)
            or report['script_sha256'] != sha256('scripts/evaluate_runtime_balanced_trial.py')
            or any(sha256(p) != h for p,h in report['helpers_sha256'].items())
            or any(sha256(p) != h for p,h in report['reviews_sha256'].items())):
        raise ValueError('Scene evaluation changed')
    for identity in report['classifiers'].values():
        path = Path(identity['path'])
        if sha256(path) != identity['sha256'] or sha256(path.with_suffix('.json')) != identity['metadata_sha256']:
            raise ValueError('Classifier changed')
    scenes = read_json(RAW)
    for panel,variants in scenes.items():
        for name,result in variants.items():
            saved = report['panels'][panel][name]
            m = checked_metrics(result)
            compared = compare_frames(result['frames'],variants['parent']['frames'])
            if (m != saved['metrics'] or any(saved[k] != v for k,v in compared.items())
                    or saved['meets_94'] != meets_target(m)
                    or saved['nonregression_vs_parent'] != nonregression(variants['parent']['metrics'],m)):
                raise ValueError('Metrics or target identity decision changed')
            if name.startswith('admit') and saved['nonregression_vs_control'] != nonregression(
                    variants['control'+name[-2:]]['metrics'],m):
                raise ValueError('Control comparison changed')
            for row,base in zip(result['frames'],variants['parent']['frames'],strict=True):
                def boxes(f):
                    return [(d['box'],d['crop_box']) for d in f['detections']]
                if boxes(row) != boxes(base) or row['neural_calls'] != base['neural_calls']:
                    raise ValueError('Classifiers did not receive the same proposal workload')
    candidates = {}
    for name in ('admit47','admit48'):
        values = [p[name] for p in report['panels'].values()]
        candidates[name] = dict(all_panels_nonregression=all(v['nonregression_vs_parent'] for v in values),
            all_parent_targets_retained=all(not v['lost_parent_targets'] for v in values),
            all_panels_94=all(v['meets_94'] for v in values),
            all_panels_nonregression_vs_control=all(v['nonregression_vs_control'] for v in values))
        candidates[name]['eligible_for_pi_comparison'] = (
            candidates[name]['all_panels_nonregression'] and candidates[name]['all_parent_targets_retained'])
    write_json(OUTPUT,dict(candidates=candidates,panels=report['panels'],
        inputs={str(p):sha256(p) for p in (TRIAL,DEVELOPMENT,RAW)},script_sha256=sha256(__file__),
        decision='Pi comparison may be considered; independent qualification still required' if any(
            c['eligible_for_pi_comparison'] for c in candidates.values()) else
            'Keep prior research classifier: no admitted seed passes scene nonregression and target-retention gates.',
        scope='Fixed paired TRAIN intervention; related repeated development panels are not independent evidence of94%or flight readiness. Parameter count and call count unchanged, speed unmeasured.',
        pi_tested=False,test_evaluated=False,deployment_approved=False,qualification_achieved=False))
    print(candidates)


if __name__ == '__main__':
    summarize()
