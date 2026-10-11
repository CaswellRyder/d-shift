"""Independent count/identity recount of the paired synthetic color trial."""
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from scripts.audit_balloon_hue import OUTPUT as AUDIT
from scripts.diagnose_clutter_stage_attribution import difference
from scripts.summarize_public_balloon_trial import checked_metrics
from scripts.balloon_scene_constraints import decisions

OUTPUT = Path('research/balloon-red-blue-20261008/balloon-hue-decision-20261011.json')


def summarize():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    audit = read_json(AUDIT)
    if (sha256(audit['raw']['path']) != audit['raw']['sha256']
            or any(sha256(p) != h for p,h in audit['inputs_sha256'].items())
            or any(audit[k] is not False for k in ('test_evaluated','development_evaluated',
                                                 'deployment_approved','pi_tested','qualification_achieved'))):
        raise ValueError('Audit source or scope changed')
    raw = read_json(audit['raw']['path'])
    names = {'control47','control48','hue47','hue48'}
    if set(raw) != names|{'parent'} or set(audit['results']) != names:
        raise ValueError('Incomplete paired trial')
    results = {}
    for name in sorted(names):
        row = audit['results'][name]
        root = Path(row['run'])
        if any(sha256(root/f) != h for f,h in row['artifacts_sha256'].items()):
            raise ValueError('Changed run artifact')
        prov = read_json(root/'provenance.json')
        if any(sha256(p) != h for p,h in prov['sources_sha256'].items()):
            raise ValueError('Changed training source')
        metrics = {k:checked_metrics(raw[name][k]) for k in ('original','window1')}
        effects = {k:difference(raw[name][k],raw['parent'][k]) for k in metrics}
        with np.load(root/'predictions.npz', allow_pickle=False) as stored:
            negatives = [int(np.count_nonzero(decisions(stored[k]))) for k in ('probe_before','probe_after')]
        retained = all(not e['lost_targets'] and all(v['tp']>=0 and v['fp']<=0
                       for v in e['count_delta'].values()) for e in effects.values())
        improved = any(v['tp']>0 or v['fp']<0 for e in effects.values() for v in e['count_delta'].values())
        gate = retained and improved and negatives[1]<=negatives[0]
        if (metrics['original'] != row['original_metrics'] or metrics['window1'] != row['window1_metrics']
                or effects['original'] != row['effect'] or effects['window1'] != row['new_effect']
                or negatives != row['background_false_acceptances'] or gate != row['training_scene_gate_pass']):
            raise ValueError('Detection recount disagrees')
        results[name] = dict(metrics=metrics,effects=effects,background_false_acceptances=negatives,
                             training_scene_gate_pass=gate)
        if name.startswith('hue'):
            contrast = {k:difference(raw[name][k],raw['control'+name[-2:]][k]) for k in metrics}
            if contrast != row['against_matched_control']:
                raise ValueError('Paired contrast disagrees')
            results[name]['against_matched_control'] = contrast
    eligible = all(results[f'hue{s}']['training_scene_gate_pass'] for s in (47,48))
    if eligible != audit['eligible_for_export_and_development']:
        raise ValueError('Promotion decision disagrees')
    write_json(OUTPUT, dict(audit_sha256=sha256(AUDIT),results=results,
        inputs_sha256={p:sha256(p) for p in (__file__,'scripts/diagnose_clutter_stage_attribution.py',
            'scripts/summarize_public_balloon_trial.py','scripts/balloon_scene_constraints.py')},
        eligible_for_export_and_development=eligible,
        decision='Eligible only for separate export and development evaluation.' if eligible else
                 'Reject hue intervention: paired TRAIN retention/improvement gate failed. Pi unchanged.',
        scope='TRAIN-only full-scene decision recount on fixed classifier-independent proposals, not generalization or flight evidence.',
        development_evaluated=False,test_evaluated=False,pi_tested=False,
        deployment_approved=False,qualification_achieved=False))
    print(dict(eligible=eligible,results=results),flush=True)


if __name__ == '__main__':
    summarize()
