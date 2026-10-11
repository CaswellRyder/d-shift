"""Independently recount stage effects without selecting a failed model."""
from pathlib import Path

from dtr.data import read_json, sha256, write_json
from scripts.diagnose_clutter_stage_attribution import OUTPUT as SOURCE, PROTOCOL, difference
from scripts.summarize_public_balloon_trial import checked_metrics

OUTPUT=Path('research/balloon-red-blue-20261008/clutter-stage-attribution-decision-20261011.json')


def summarize():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    doc=read_json(SOURCE)
    if (doc['script_sha256']!=sha256('scripts/diagnose_clutter_stage_attribution.py')
            or doc['protocol_sha256']!=sha256(PROTOCOL)
            or sha256(doc['raw']['path'])!=doc['raw']['sha256']
            or any(sha256(k)!=h for k,h in {**doc['helpers_sha256'],**doc['inputs']}.items())
            or any(sha256(r['path'])!=r['sha256'] for r in doc['models'].values())):
        raise ValueError('Attribution evidence changed')
    raw=read_json(doc['raw']['path'])
    p=read_json(PROTOCOL)
    if set(raw)!=set(p['panels']) or set(raw)!=set(doc['panels']):
        raise ValueError('Missing panel')
    summary={}
    for panel,variants in raw.items():
        if set(variants)!=set(p['variants']):
            raise ValueError('Missing matched stage')
        summary[panel]={}
        for name,result in variants.items():
            expected=dict(metrics=checked_metrics(result),against_parent=difference(result,variants['parent']))
            if name.startswith('corrected/'):
                expected['correction_effect']=difference(result,variants['feature/'+name.split('/')[1]])
            if 'admit' in name:
                expected['new_data_effect']=difference(result,variants[name.replace('admit','control')])
            if expected!=doc['panels'][panel][name]:
                raise ValueError('Counts or identity-level attribution do not reproduce')
            summary[panel][name]=dict(metrics=expected['metrics'],
                effects={k:dict(count_delta=v['count_delta'],lost=len(v['lost_targets']),gained=len(v['gained_targets']))
                         for k,v in expected.items() if k!='metrics'})
    write_json(OUTPUT,dict(source_sha256=sha256(SOURCE),script_sha256=sha256(__file__),panels=summary,
        finding='Both negative admission and forced correction hurt recall. On67-scene DEV, each correction loses a further4red and6blue true positives relative to its own uncorrected feature model. This is a controlled post-hoc attribution, not independent qualification.',
        decision='Do not deploy any of these models or continue forcing this13crop fit. Test input context on TRAIN scenes before further architecture or optimizer changes.',
        training_performed=False,development_evaluated=True,test_evaluated=False,pi_tested=False,
        deployment_approved=False,qualification_achieved=False))
    print('Stage effects reproduced; forced correction rejected. Next: TRAIN-only context diagnostic.',flush=True)


if __name__=='__main__':
    summarize()
