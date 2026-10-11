"""Recount gradient evidence; suppress interpretation of negligible task gradients."""
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from scripts.diagnose_balloon_context_gradients import OUTPUT as SOURCE, GROUPS
from scripts.balloon_gradient_summary import summarize

OUTPUT=Path('research/balloon-red-blue-20261008/balloon-context-gradient-decision-20261011.json')


def audit():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    doc=read_json(SOURCE)
    if (doc['script_sha256']!=sha256('scripts/diagnose_balloon_context_gradients.py')
            or sha256(doc['archive']['path'])!=doc['archive']['sha256']
            or any(sha256(k)!=h for k,h in doc['helpers_sha256'].items())
            or any(sha256(v['path'])!=v['sha256'] for v in doc['models'].values())
            or any(doc[k] is not False for k in ('weights_changed','training_performed','development_evaluated',
                                                'test_evaluated','pi_tested','deployment_approved','qualification_achieved'))):
        raise ValueError('Gradient diagnostic changed')
    counts={k:b-a for k,(a,b) in GROUPS.items()}
    results={}
    with np.load(doc['archive']['path'],allow_pickle=False) as saved:
        for name,row in doc['results'].items():
            gradients={k:saved[name+'/'+k+'/gradient'] for k in GROUPS}
            measured=summarize(gradients,counts)
            objective=0.
            for group,count in counts.items():
                values=saved[name+'/'+group+'/losses']
                expected=dict(rows=count,weighted_mean_loss=float(np.mean(values)),
                              contribution_to_total=float(np.sum(values)/sum(counts.values())))
                if values.shape!=(count,) or expected!=row['groups'][group]:
                    raise ValueError('Group objective does not reproduce')
                objective+=expected['contribution_to_total']
            if measured!=row['gradients'] or objective!=row['objective']:
                raise ValueError('Gradient summary does not reproduce')
            norms=measured['contribution_norms']
            active={g for g,n in norms.items() if n>max(norms.values())*1e-6}
            results[name]=dict(objective=objective,cancellation_ratio=measured['cancellation_ratio'],
                meaningful_pairs={k:v for k,v in measured['pairs'].items() if all(g in active for g in k.split('/'))})
    write_json(OUTPUT,dict(source_sha256=sha256(SOURCE),script_sha256=sha256(__file__),results=results,
        finding='Both context checkpoints exhibit strong local conflict between old-crop and scene objectives. This does not prove additional epochs or model capacity cannot help.',
        next_experiment='Matched two-task gradient projection versus ordinary summed gradients on the same wider-context TRAIN tensors, seeds, batches and optimizer schedule. No inference architecture change.',
        primary_reference='https://arxiv.org/abs/2001.06782',training_performed=False,
        development_evaluated=False,test_evaluated=False,pi_tested=False,deployment_approved=False,qualification_achieved=False))
    print(results,flush=True)


if __name__=='__main__':
    audit()
