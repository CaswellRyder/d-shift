"""Independently recount context TRAIN scene eligibility; retain failed models as research."""
from pathlib import Path

from dtr.data import read_json, sha256, write_json
from scripts.audit_balloon_gradient_projection import OUTPUT as AUDIT, PROTOCOL
from scripts.audit_balloon_context_distillation import scene_gate, false_acceptances
import numpy as np
from scripts.diagnose_clutter_stage_attribution import difference
from scripts.summarize_public_balloon_trial import checked_metrics

OUTPUT=Path('research/balloon-red-blue-20261008/balloon-gradient-projection-decision-20261011.json')


def summarize():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    doc=read_json(AUDIT)
    if (doc['script_sha256']!=sha256('scripts/audit_balloon_gradient_projection.py')
            or doc['protocol_sha256']!=sha256(PROTOCOL)
            or sha256(doc['raw']['path'])!=doc['raw']['sha256']
            or any(doc[k] is not False for k in ('development_evaluated','test_evaluated',
                                                'pi_tested','deployment_approved','qualification_achieved'))):
        raise ValueError('Context audit or scope changed')
    raw=read_json(doc['raw']['path'])
    if set(raw)!={'parent','sum47','sum48','project47','project48'}:
        raise ValueError('Missing matched scene result')
    baseline=checked_metrics(raw['parent'])
    results={}
    for name,row in doc['results'].items():
        root=Path(row['run'])
        prov=read_json(root/'provenance.json')
        if any(sha256(k)!=h for k,h in prov['sources_sha256'].items()):
            raise ValueError('Training sources changed')
        for key,file in (('model_sha256','student.keras'),('predictions_sha256','predictions.npz'),
                         ('report_sha256','report.json'),('provenance_sha256','provenance.json'),
                         ('trace_sha256','trace.json')):
            if sha256(root/file)!=row[key]:
                raise ValueError('Training artifact changed')
        measured=checked_metrics(raw[name])
        effect=difference(raw[name],raw['parent'])
        with np.load(root/'predictions.npz',allow_pickle=False) as stored:
            negatives=[false_acceptances(stored[k]) for k in ('probe_before','probe_after')]
        if negatives!=row['background_false_acceptances']:
            raise ValueError('Negative acceptance counts do not reproduce')
        gate=scene_gate(effect,*negatives)
        if measured!=row['metrics'] or effect!=row['effect'] or gate!=row['training_scene_gate_pass']:
            raise ValueError('Context scene counts/gate do not reproduce')
        results[name]=dict(metrics=measured,lost=len(effect['lost_targets']),gained=len(effect['gained_targets']),
            background_false_acceptances=row['background_false_acceptances'],training_scene_gate_pass=gate)
        if name.startswith('project'):
            against=difference(raw[name],raw['sum'+name[-2:]])
            results[name]['against_matched_control']=dict(count_delta=against['count_delta'],
                lost=len(against['lost_targets']),gained=len(against['gained_targets']))
    passed=all(results[f'project{s}']['training_scene_gate_pass'] for s in (47,48))
    if passed!=doc['eligible_for_export_and_development']:
        raise ValueError('Paired decision changed')
    write_json(OUTPUT,dict(audit_sha256=sha256(AUDIT),script_sha256=sha256(__file__),
        parent_metrics=baseline,results=results,eligible_for_export_and_development=passed,
        decision='Projected pair eligible for export parity and repeatedDEV, not deployment.' if passed else 'Reject projected-gradient pair: TRAIN scene nonregression not achieved.',
        next_step='Use the measured matched-control effects to decide whether optimization is sufficient; do not force a post-hoc perfect-crop correction or deploy failed models.',
        scope='Matched summed versus projected gradient updates; same wider-context data and teacher, both seeds. All scene counts and saved negative-probe decisions reproduced. Training data only; no Pi or flight qualification.',
        development_evaluated=False,test_evaluated=False,pi_tested=False,deployment_approved=False,qualification_achieved=False))
    print(dict(eligible_for_export_and_development=passed,results=results),flush=True)


if __name__=='__main__':
    summarize()

