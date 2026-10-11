"""Independently recount TRAIN detections before admitting a window1 candidate to DEV."""
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from scripts.audit_window1_runtime import OUTPUT as AUDIT
from scripts.window1_trial_data import tensors, recovered_targets
from scripts.audit_balloon_context_distillation import scene_gate, false_acceptances
from scripts.diagnose_clutter_stage_attribution import difference
from scripts.summarize_public_balloon_trial import checked_metrics

OUTPUT = Path('research/balloon-red-blue-20261008/window1-runtime-decision-20261011.json')


def summarize():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    audit = read_json(AUDIT)
    if (sha256(audit['raw']['path']) != audit['raw']['sha256']
            or any(sha256(p) != h for p, h in audit['inputs_sha256'].items())
            or any(audit[k] is not False for k in ('development_evaluated', 'test_evaluated',
                                                 'pi_tested', 'deployment_approved', 'qualification_achieved'))):
        raise ValueError('Changed audit or scope')
    raw = read_json(audit['raw']['path'])
    names = {'control47', 'control48', 'hard47', 'hard48'}
    if set(raw) != names | {'parent'} or set(audit['results']) != names:
        raise ValueError('Incomplete matched experiment')
    results = {}
    for name, row in audit['results'].items():
        root = Path(row['run'])
        if any(sha256(root/f) != h for f, h in row['artifacts_sha256'].items()):
            raise ValueError('Run artifacts changed')
        prov = read_json(root/'provenance.json')
        if any(sha256(p) != h for p, h in prov['sources_sha256'].items()):
            raise ValueError('Training source changed')
        old, new = raw[name]['original'], raw[name]['window1']
        old_metrics, new_metrics = checked_metrics(old), checked_metrics(new)
        effect, new_effect = difference(old, raw['parent']['original']), difference(new, raw['parent']['window1'])
        with np.load(root/'predictions.npz', allow_pickle=False) as stored:
            negatives = [false_acceptances(stored[k]) for k in ('probe_before', 'probe_after')]
        probabilities = np.asarray(row['batch_probabilities'])[-2:]
        hard_correct = (probabilities.argmax(1) == 2) & (probabilities[:, 2] >= .8)
        recovered = recovered_targets(new)
        gate = bool(scene_gate(effect, *negatives) and scene_gate(new_effect, *negatives)
                    and hard_correct.all() and recovered)
        if (old_metrics != row['original_metrics'] or new_metrics != row['window1_metrics']
                or effect != row['effect'] or new_effect != row['new_effect']
                or negatives != row['background_false_acceptances']
                or hard_correct.tolist() != row['batch_correct'][-2:]
                or gate != row['training_scene_gate_pass']):
            raise ValueError('Training decision does not reproduce')
        results[name] = dict(original_metrics=old_metrics, window1_metrics=new_metrics,
            targeted_blue_recovered=recovered, runtime_blue_probabilities=probabilities[:, 2].tolist(),
            lost_original_targets=effect['lost_targets'], lost_window1_targets=new_effect['lost_targets'],
            gained_window1_targets=new_effect['gained_targets'], background_false_acceptances=negatives,
            training_scene_gate_pass=gate)
        if name.startswith('hard'):
            contrast = {k:difference(raw[name][k], raw['control'+name[-2:]][k]) for k in ('original', 'window1')}
            if contrast != row['against_matched_control']:
                raise ValueError('Same-seed comparison changed')
            results[name]['against_matched_control'] = contrast
    data = tensors('hard')
    conflicts = []
    for runtime_id, pixels in enumerate(data['batch_pixels'][-2:]):
        same = np.flatnonzero(np.all(data['x'][:2126] == pixels, axis=(1, 2, 3)))
        conflicts.append(dict(runtime_id=runtime_id, exact_old_pixel_duplicates=same.tolist(),
            conflicting_labels=[int(i) for i in same if data['y'][i] != 2]))
    eligible = all(results[f'hard{s}']['training_scene_gate_pass'] for s in (47, 48))
    if eligible != audit['eligible_for_export_and_development']:
        raise ValueError('Paired promotion decision changed')
    write_json(OUTPUT, dict(audit_sha256=sha256(AUDIT), inputs_sha256={p:sha256(p) for p in (
        __file__, 'scripts/window1_trial_data.py', 'scripts/audit_current_blue_search.py',
        'scripts/diagnose_clutter_stage_attribution.py', 'scripts/summarize_public_balloon_trial.py',
        'scripts/audit_balloon_context_distillation.py')}, results=results, conflict_checks=conflicts,
        eligible_for_export_and_development=eligible,
        decision='Eligible for separate parity and full-search development evaluation.' if eligible else
                 'Reject both hard candidates; no model promotion or threshold change.',
        scope='TRAIN-only matched experiment. Scene proposal localization limitations remain. No independent generalization or Pi speed claim.',
        development_evaluated=False, test_evaluated=False, pi_tested=False,
        deployment_approved=False, qualification_achieved=False))
    print(dict(eligible=eligible, conflicts=conflicts, results=results), flush=True)


if __name__ == '__main__':
    summarize()
