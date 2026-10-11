"""Independently recount decision-retention results and reviewed-label learning."""
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from dtr.runtime import softmax
from scripts.scene_teacher_supervision import REVIEW
from scripts.scene_decision_retention import tensors, PROTOCOL
from scripts.audit_scene_decision_retention import OUTPUT as AUDIT
from scripts.audit_balloon_context_distillation import scene_gate, false_acceptances
from scripts.diagnose_clutter_stage_attribution import difference
from scripts.summarize_public_balloon_trial import checked_metrics

OUTPUT = Path('research/balloon-red-blue-20261008/scene-decision-retention-decision-20261011.json')


def summarize():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    doc = read_json(AUDIT)
    if (sha256(PROTOCOL) != doc['protocol_sha256'] or sha256(REVIEW) != doc['review_sha256']
            or sha256('scripts/audit_scene_decision_retention.py') != doc['script_sha256']
            or sha256(doc['raw']['path']) != doc['raw']['sha256']
            or any(doc[k] is not False for k in ('development_evaluated', 'test_evaluated',
                                                'pi_tested', 'deployment_approved', 'qualification_achieved'))):
        raise ValueError('Audit identity or scope changed')
    raw, data = read_json(doc['raw']['path']), tensors()
    names = {'control47', 'control48', 'decision47', 'decision48'}
    if set(raw) != names | {'parent'} or set(doc['results']) != names:
        raise ValueError('Missing matched result')
    ids = np.flatnonzero(data['correction_labels'] >= 0)
    exact = []
    for i in ids:
        same = np.flatnonzero(np.all(data['x'][:2126] == data['x'][i], axis=(1, 2, 3)))
        exact.append(dict(proposal_index=int(i-2126), prior_pixel_duplicates=same.tolist(),
                          conflicting_hard_labels=[int(j) for j in same if data['y'][j] != data['correction_labels'][i]]))
    results = {}
    for name, row in doc['results'].items():
        root = Path(row['run'])
        for key, file in (('model_sha256', 'student.keras'), ('report_sha256', 'report.json'),
                          ('provenance_sha256', 'provenance.json'), ('predictions_sha256', 'predictions.npz'),
                          ('epochs_sha256', 'epochs.csv')):
            if sha256(root/file) != row[key]:
                raise ValueError('Training artifact changed')
        prov = read_json(root/'provenance.json')
        if any(sha256(p) != h for p, h in prov['sources_sha256'].items()):
            raise ValueError('Training source changed')
        measured, effect = checked_metrics(raw[name]), difference(raw[name], raw['parent'])
        with np.load(root/'predictions.npz', allow_pickle=False) as stored:
            negatives = [false_acceptances(stored[k]) for k in ('probe_before', 'probe_after')]
            before, after = [softmax(stored[k][ids]) for k in ('before', 'after')]
        target = data['correction_labels'][ids]
        correct = ((after.argmax(1) == target) & ((target == 0) | (after.max(1) >= .8))).tolist()
        gate = scene_gate(effect, *negatives)
        if (measured != row['metrics'] or effect != row['effect'] or correct != row['review_correct']
                or negatives != row['background_false_acceptances'] or gate != row['training_scene_gate_pass']):
            raise ValueError('Outcome does not reproduce')
        results[name] = dict(metrics=measured, lost_targets=effect['lost_targets'],
            background_false_acceptances=negatives, training_scene_gate_pass=gate,
            reviewed_correct=sum(correct), reviewed_probabilities=[dict(proposal_index=int(i-2126),
                label=int(c), before=float(b[c]), after=float(a[c]), correct=ok)
                for i, c, b, a, ok in zip(ids, target, before, after, correct, strict=True)])
        if name.startswith('decision'):
            results[name]['against_matched_control'] = difference(raw[name], raw['control'+name[-2:]])
    eligible = all(results[f'decision{s}']['training_scene_gate_pass'] for s in (47, 48))
    if eligible != doc['eligible_for_export_and_development']:
        raise ValueError('Paired gate changed')
    write_json(OUTPUT, dict(audit_sha256=sha256(AUDIT), script_sha256=sha256(__file__),
        results=results, exact_pixel_conflict_check=exact, eligible_for_export_and_development=eligible,
        decision='Eligible only for separate export and repeatedDEV.' if eligible else 'Reject both decision-retention candidates; full TRAIN detection retention failed.',
        scope='Ten reviewed crops from existing TRAIN scenes, correlated and not independent additions. Exact-pixel check does not rule out near-duplicate or other semantic label conflicts. Related development and reserved test untouched.',
        development_evaluated=False, test_evaluated=False, pi_tested=False,
        deployment_approved=False, qualification_achieved=False))
    print(dict(eligible=eligible, exact_pixel_conflicts=sum(len(r['conflicting_hard_labels']) for r in exact),
        reviewed={n:dict(correct=r['reviewed_correct'], improved_probability=sum(p['after']>p['before'] for p in r['reviewed_probabilities']))
                  for n, r in results.items()}), flush=True)


if __name__ == '__main__':
    summarize()

