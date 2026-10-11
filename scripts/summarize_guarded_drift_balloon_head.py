"""Recount deterministic constrained-head scenes; validate retention and fixed workload."""
from pathlib import Path

from dtr.data import read_json, sha256, write_json
from scripts.audit_indoor5_outcomes import compare_frames
from scripts.evaluate_guarded_drift_balloon_head import OUTPUT as SCENES, BASELINE, development_names
from scripts.evaluate_current_confident_containment import nonregressing
from scripts.evaluate_red_blue_development import LABELS
from scripts.export_guarded_drift_balloon_head import checked_trial
from scripts.summarize_dense_background_trial import check_baseline
from scripts.summarize_public_balloon_trial import checked_metrics

OUTPUT = Path('research/balloon-red-blue-20261008/guarded-drift-balloon-head-decision-20261011.json')


def audit():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    _, names = checked_trial()
    doc, reference = read_json(SCENES), read_json(BASELINE)
    if (sha256(doc['raw']['path']) != doc['raw']['sha256']
            or sha256(BASELINE) != doc['baseline_sha256']
            or sha256(reference['raw']['path']) != reference['raw']['sha256']
            or sha256('scripts/evaluate_guarded_drift_balloon_head.py') != doc['script_sha256']
            or any(sha256(p) != h for p, h in {**doc['helpers_sha256'], **doc['inputs']}.items())):
        raise ValueError('Scene evaluation changed')
    for export in doc['exports'].values():
        path = Path(export['path'])
        if sha256(path) != export['sha256'] or sha256(path.with_suffix('.json')) != export['metadata_sha256']:
            raise ValueError('Export identity changed')
    raw, baseline = read_json(doc['raw']['path']), read_json(reference['raw']['path'])
    if set(raw) != set(doc['panels']) or 'training' not in raw:
        raise ValueError('Missing scene comparison')
    summaries = {}
    for panel, variants in raw.items():
        if set(variants) != {'parent', *names}:
            raise ValueError('Incomplete matched scene comparison')
        parent = variants['parent']['frames']
        for a, b in zip(parent, baseline[panel]['frames'], strict=True):
            check_baseline(a, b)
        summaries[panel] = {}
        for name, result in variants.items():
            measured = checked_metrics(result)
            compared = compare_frames(result['frames'], parent)
            gate = nonregressing(variants['parent']['metrics'], measured, compared)
            expected = dict(metrics=measured, comparison=compared, nonregression=gate,
                meets_94=all(measured[c][m] >= .94 for c in LABELS for m in ('precision', 'recall')),
                neural_calls=sum(f['neural_calls'] for f in result['frames']))
            if expected != doc['panels'][panel][name]:
                raise ValueError('Scene summary does not reproduce')
            for a, b in zip(result['frames'], parent, strict=True):
                if a['neural_calls'] != b['neural_calls']:
                    raise ValueError('Classifier comparison changed call count')
                for x, y in zip(a['detections'], b['detections'], strict=True):
                    if any(x[k] != y[k] for k in ('box', 'crop_box')):
                        raise ValueError('Classifier comparison changed proposal geometry')
            summaries[panel][name] = dict(metrics=measured, nonregression=gate,
                lost=len(compared['lost_parent_targets']), gained=len(compared['gained_targets']),
                errors={c: compared['colors'][c]['errors'] for c in LABELS})
            if name.startswith('admit'):
                control = variants['control']
                summaries[panel][name]['against_matched_control'] = compare_frames(result['frames'], control['frames'])
    allowed = development_names(names, doc['panels']['training'])
    if allowed != doc['eligible_for_development'] or (set(raw) != (set(baseline) if allowed else {'training'})):
        raise ValueError('Unexpected development evaluation scope')
    viable = [n for n in allowed if n.startswith('admit') and len(raw) == 4
              and all(p[n]['nonregression'] for p in doc['panels'].values())]
    if (viable != doc['eligible_for_pi_comparison'] or doc['development_evaluated'] != (len(raw)>1)
            or any(doc[k] is not False for k in ('test_evaluated', 'pi_tested', 'deployment_approved', 'qualification_achieved'))):
        raise ValueError('Unsupported promotion or scope claim')
    result = dict(scene_receipt_sha256=sha256(SCENES), script_sha256=sha256(__file__),
        summaries=summaries, eligible_for_pi_comparison=viable,
        decision='Eligible for separate Pi research comparison, not deployment.' if viable else
                 'Do not promote these classifiers; the constrained admission fails scene nonregression.',
        scope='Recounted full-scene TRAIN and any related repeated development. Deterministic unchanged control isolates the new-data intervention; related panels must not be pooled as independent accuracy. No Pi timing or qualification.',
        test_evaluated=False, pi_tested=False, deployment_approved=False, qualification_achieved=False)
    write_json(OUTPUT, result)
    print(result['decision'], flush=True)
    for panel, variants in summaries.items():
        print(panel, {n: {k: r[k] for k in ('metrics', 'lost', 'gained')} for n, r in variants.items()}, flush=True)


if __name__ == '__main__':
    audit()



