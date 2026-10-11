"""Independent count/identity recount of the paired real-data expansion trial."""
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from scripts.audit_window2_retraining import OUTPUT as AUDIT
from scripts.diagnose_clutter_stage_attribution import difference
from scripts.summarize_public_balloon_trial import checked_metrics
from scripts.balloon_scene_constraints import decisions
from scripts.audit_blue_balloon_window2 import checked_extra

OUTPUT = Path('research/balloon-red-blue-20261008/window2-retraining-decision-20261011.json')


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
    names = {'control47','control48','admit47','admit48'}
    if set(raw) != names|{'parent'} or set(audit['results']) != names:
        raise ValueError('Incomplete paired trial')
    results = {}
    labels = {r['id']:('background','red_balloon','blue_balloon').index(r['label']) for r in checked_extra()}
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
        probabilities = np.asarray(row['window2_probabilities'])
        ids = row['window2_ids']
        if len(ids)!=15 or set(ids)!=set(labels) or probabilities.shape!=(15,3) or not np.isfinite(probabilities).all():
            raise ValueError('Invalid new-crop prediction inventory')
        expected = np.array([labels[i] for i in ids])
        passed = (probabilities.argmax(1)==expected)&(probabilities[np.arange(15),expected]>=.8)
        if passed.tolist()!=row['window2_correct']:
            raise ValueError('New-crop correctness differs')
        gate = bool(retained and passed.all() and negatives[1]<=negatives[0])
        if (metrics['original'] != row['original_metrics'] or metrics['window1'] != row['window1_metrics']
                or effects['original'] != row['effect'] or effects['window1'] != row['new_effect']
                or negatives != row['background_false_acceptances'] or gate != row['training_scene_gate_pass']):
            raise ValueError('Detection recount disagrees')
        results[name] = dict(metrics=metrics,effects=effects,background_false_acceptances=negatives,
                             training_scene_gate_pass=gate,
            window2_correct=int(passed.sum()), window2_misses=[dict(id=ids[i],scores=probabilities[i].tolist())
                for i in np.flatnonzero(~passed)])
        if name.startswith('admit'):
            contrast = {k:difference(raw[name][k],raw['control'+name[-2:]][k]) for k in metrics}
            if contrast != row['against_matched_control']:
                raise ValueError('Paired contrast disagrees')
            results[name]['against_matched_control'] = contrast
    eligible = all(results[f'admit{s}']['training_scene_gate_pass'] for s in (47,48))
    if eligible != audit['eligible_for_window2_scene_audit']:
        raise ValueError('Promotion decision disagrees')
    write_json(OUTPUT, dict(audit_sha256=sha256(AUDIT),results=results,
        inputs_sha256={p:sha256(p) for p in (__file__,'scripts/diagnose_clutter_stage_attribution.py',
            'scripts/summarize_public_balloon_trial.py','scripts/balloon_scene_constraints.py')},
        eligible_for_window2_scene_audit=eligible,
        decision='Eligible only for window2 complete-scene annotation and TRAIN detector audit.' if eligible else
                 'Reject current window2 models: paired scene retention/new-crop recovery gate failed. Pi unchanged.',
        scope='TRAIN-only original64/window1six-scene recount and15window2crop decisions. Twelve window2photos still require complete target annotations and detector audit; crop results are not scene precision/recall or held-out/flight evidence.',
        development_evaluated=False,test_evaluated=False,pi_tested=False,
        deployment_approved=False,qualification_achieved=False))
    print(dict(eligible=eligible,results=results),flush=True)


if __name__ == '__main__':
    summarize()
