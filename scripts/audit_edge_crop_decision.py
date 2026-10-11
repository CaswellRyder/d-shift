"""Recount paired crop retraining and factor swaps before rejecting promotion."""
from pathlib import Path

from dtr.data import read_json, sha256, write_json
from scripts.audit_indoor5_outcomes import compare_frames
from scripts.diagnose_edge_crop_localizer import OUTPUT as ATTRIBUTION
from scripts.evaluate_edge_crop_localizer import OUTPUT as TRIAL
from scripts.summarize_dense_background_trial import check_baseline
from scripts.summarize_public_balloon_trial import checked_metrics

OUTPUT = Path('research/balloon-red-blue-20261008/edge-crop-decision-20261011.json')


def audit_results(trial, raw, attribution, factors):
    """Validate every row, identity-level comparison and claimed summary."""
    names = {f'{arm}{seed}' for arm in ('control', 'edge_crop') for seed in (47, 48)}
    if set(raw) != names | {'parent'} or set(trial['training_scenes']) != set(raw):
        raise ValueError('Incomplete paired trial')
    expected = {f'{name}/{part}' for name in names for part in ('both', 'heat_only', 'geometry_only')}
    if set(factors) != expected or set(attribution['results']) != expected:
        raise ValueError('Incomplete factor comparison')
    summaries = {}
    for name, result in {**raw, **factors}.items():
        frames = result['frames']
        if len(frames) != 64 or len({f['source'] for f in frames}) != 64:
            raise ValueError('Incomplete or duplicate TRAIN scenes')
        measured = checked_metrics(result)
        compared = compare_frames(frames, raw['parent']['frames'])
        declared = (trial['training_scenes'] if name in raw else attribution['results'])[name]
        if measured != declared['metrics'] or compared != declared['comparison']:
            raise ValueError('Claimed metrics or target identities changed')
        summaries[name] = dict(metrics=measured, lost=len(compared['lost_parent_targets']),
                               gained=len(compared['gained_targets']))
    for name in names:
        for a, b in zip(raw[name]['frames'], factors[name+'/both']['frames'], strict=True):
            check_baseline(a, b)
    for doc in (trial, attribution):
        if any(doc[k] is not False for k in ('development_evaluated', 'test_evaluated',
                                            'deployment_approved', 'qualification_achieved')):
            raise ValueError('Unsupported evaluation or qualification scope')
    if trial['eligible_for_development'] is not False:
        raise ValueError('Unexpected promotion claim')
    # This receipt states a failed experiment, not a generic positive-result gate.
    for name in names:
        if not summaries[name]['lost'] or summaries[name]['gained']:
            raise ValueError('Evidence no longer supports the rejection decision')
    return summaries


def audit():
    trial, attribution = read_json(TRIAL), read_json(ATTRIBUTION)
    for doc, script in ((trial, 'scripts/evaluate_edge_crop_localizer.py'),
                        (attribution, 'scripts/diagnose_edge_crop_localizer.py')):
        if sha256(doc['raw']['path']) != doc['raw']['sha256'] or sha256(script) != doc['script_sha256']:
            raise ValueError('Changed raw outputs or producing script')
    if attribution['trial_sha256'] != sha256(TRIAL):
        raise ValueError('Attribution used a different trial')
    for export in trial['exports'].values():
        path = Path(export['path'])
        meta = read_json(path.with_suffix('.json'))
        report = read_json(path.parent/'report.json')
        provenance = read_json(path.parent/'provenance.json')
        if (meta != export['metadata'] or sha256(path) != meta['sha256']
                or report != export['training_report']
                or read_json(path.parent/'status.json')['state'] != 'complete'
                or sha256(path.parent/'proposal.keras') != report['model_sha256']
                or sha256(path.parent/'predictions.npz') != report['predictions_sha256']
                or sha256(path.parent/'provenance.json') != report['provenance_sha256']
                or any(sha256(p) != h for p, h in provenance['sources_sha256'].items())):
            raise ValueError('Training or export identity changed')
    summaries = audit_results(trial, read_json(trial['raw']['path']),
                              attribution, read_json(attribution['raw']['path']))
    return dict(trial_sha256=sha256(TRIAL), attribution_sha256=sha256(ATTRIBUTION),
        audit_script_sha256=sha256(__file__), summaries=summaries,
        decision='Reject all four checkpoints. Do not advance them to development or Pi.',
        interpretation='Lower grouped holdout loss did not improve full-scene TRAIN detections. Both edge-trained heat-only and geometry-only factor swaps lose targets; neither isolated component is rescued by retaining the other parent component.',
        scope='Recount and provenance audit of TRAIN diagnostics only. Factor swaps are retrospective two-network diagnostics, not deployable speed or independent accuracy evidence.',
        development_evaluated=False, test_evaluated=False, pi_tested=False,
        deployment_approved=False, qualification_achieved=False)


if __name__ == '__main__':
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    result = audit()
    write_json(OUTPUT, result)
    print(result['decision'])
