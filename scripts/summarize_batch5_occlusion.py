"""Independent count-conserving paired trial decision and TRAIN conflict check."""
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from scripts.audit_batch5_occlusion import OUTPUT as AUDIT
from scripts.batch5_occlusion_data import tensors, new_scene_gate
from scripts.audit_balloon_context_distillation import scene_gate, false_acceptances
from scripts.diagnose_clutter_stage_attribution import difference
from scripts.summarize_public_balloon_trial import checked_metrics

OUTPUT = Path('research/balloon-red-blue-20261008/batch5-occlusion-decision-20261011.json')


def summarize():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    audit = read_json(AUDIT)
    if (sha256(audit['raw']['path']) != audit['raw']['sha256']
            or any(sha256(p) != h for p, h in audit['inputs_sha256'].items())
            or any(audit[k] is not False for k in ('development_evaluated', 'test_evaluated',
                                                 'pi_tested', 'deployment_approved', 'qualification_achieved'))):
        raise ValueError('Audit identity or TRAIN scope changed')
    raw = read_json(audit['raw']['path'])
    names = {'control47', 'control48', 'hard47', 'hard48'}
    if set(raw) != names | {'parent'} or set(audit['results']) != names:
        raise ValueError('Incomplete paired experiment')
    results = {}
    for name, row in audit['results'].items():
        root = Path(row['run'])
        if any(sha256(root/f) != h for f, h in row['artifacts_sha256'].items()):
            raise ValueError('Model evidence changed')
        prov = read_json(root/'provenance.json')
        if any(sha256(p) != h for p, h in prov['sources_sha256'].items()):
            raise ValueError('Training sources changed')
        old, new = raw[name]['original'], raw[name]['batch5']
        old_metrics, new_metrics = checked_metrics(old), checked_metrics(new)
        effect = difference(old, raw['parent']['original'])
        with np.load(root/'predictions.npz', allow_pickle=False) as stored:
            negatives = [false_acceptances(stored[k]) for k in ('probe_before', 'probe_after')]
        # Recount hard target from the selected scene, independent of standalone crop acceptance.
        hard_scene = next(f for f in new['frames'] if f['id'] == 17)
        from scripts.audit_current_blue_search import matched_truth
        hard_recovered = any(hard_scene['truth'][i]['annotation_id'] == 54 for i in matched_truth(hard_scene))
        gate = scene_gate(effect, *negatives) and new_scene_gate(new, row['batch_correct'][-1])
        if (old_metrics != row['original_metrics'] or new_metrics != row['batch5_metrics']
                or effect != row['effect'] or negatives != row['background_false_acceptances']
                or gate != row['training_scene_gate_pass'] or (gate and not hard_recovered)):
            raise ValueError('Scene outcomes failed independent recount')
        results[name] = dict(original_metrics=old_metrics, batch5_metrics=new_metrics,
            hard_recovered=hard_recovered, hard_blue_probability=row['hard_blue_probability'],
            lost_targets=effect['lost_targets'], background_false_acceptances=negatives,
            training_scene_gate_pass=bool(gate))
        if name.startswith('hard'):
            comparison = {k:difference(raw[name][k], raw['control'+name[-2:]][k]) for k in ('original', 'batch5')}
            if comparison != row['against_matched_control']:
                raise ValueError('Matched contrast changed')
            results[name]['against_matched_control'] = comparison
    data = tensors('hard')
    hard = data['batch_pixels'][-1]
    same = np.flatnonzero(np.all(data['x'][:2126] == hard, axis=(1, 2, 3)))
    eligible = all(results[f'hard{s}']['training_scene_gate_pass'] for s in (47, 48))
    if eligible != audit['eligible_for_export_and_development']:
        raise ValueError('Paired decision changed')
    write_json(OUTPUT, dict(audit_sha256=sha256(AUDIT), script_sha256=sha256(__file__),
        helpers_sha256={p:sha256(p) for p in ('scripts/audit_current_blue_search.py',
            'scripts/summarize_public_balloon_trial.py', 'scripts/diagnose_clutter_stage_attribution.py')},
        results=results, exact_old_pixel_duplicates=same.tolist(),
        conflicting_hard_labels=[int(i) for i in same if data['y'][i] != 2],
        eligible_for_export_and_development=eligible,
        decision='Eligible only for separate parity/full-search repeatedDEV.' if eligible else
                 'Reject both hard-crop candidates. No model promotion or threshold relaxation.',
        scope='TRAIN diagnostics; exact-pixel check does not exclude near duplicates or semantic conflicts. No independent accuracy or speed claim.',
        development_evaluated=False, test_evaluated=False, pi_tested=False,
        deployment_approved=False, qualification_achieved=False))
    print(dict(eligible=eligible, exact_old_pixel_duplicates=same.tolist(), results=results), flush=True)


if __name__ == '__main__':
    summarize()
