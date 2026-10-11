"""Recount fixed-context TRAIN ablation and enforce its unchanged decision gate."""
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from scripts.diagnose_balloon_wider_crop import OUTPUT as SOURCE, PROTOCOL
from scripts.balloon_wider_crop import wider_crop
from scripts.diagnose_clutter_stage_attribution import difference
from scripts.summarize_public_balloon_trial import checked_metrics

OUTPUT=Path('research/balloon-red-blue-20261008/balloon-wider-crop-decision-20261011.json')


def summarize():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    doc=read_json(SOURCE)
    if (doc['script_sha256']!=sha256('scripts/diagnose_balloon_wider_crop.py')
            or doc['protocol_sha256']!=sha256(PROTOCOL)
            or sha256(doc['raw']['path'])!=doc['raw']['sha256']
            or any(sha256(p)!=h for p,h in doc['helpers_sha256'].items())
            or any(doc[k] is not False for k in ('training_performed','development_evaluated','test_evaluated',
                                                'pi_tested','deployment_approved','qualification_achieved'))):
        raise ValueError('Context experiment changed')
    raw=read_json(doc['raw']['path'])
    if set(raw)!={'original','wider'} or any(len(v['frames'])!=64 for v in raw.values()):
        raise ValueError('Unexpected TRAIN coverage')
    measured={n:checked_metrics(r) for n,r in raw.items()}
    effect=difference(raw['wider'],raw['original'])
    if measured!=doc['metrics'] or effect!=doc['effect']:
        raise ValueError('Context metrics do not reproduce')
    for old,new in zip(raw['original']['frames'],raw['wider']['frames'],strict=True):
        if old['neural_calls']!=new['neural_calls']:
            raise ValueError('Additional neural calls')
        for a,b in zip(old['detections'],new['detections'],strict=True):
            if a['box']!=b['box'] or wider_crop(a['crop_box'],(240,320,3))!=b['crop_box']:
                raise ValueError('Context geometry changed')
    if len(doc['probes'])!=13 or len({r['id'] for r in doc['probes']})!=13:
        raise ValueError('Incomplete negative probes')
    for r in doc['probes']:
        scores=np.asarray(r['scores'])
        if (scores.shape!=(2,3) or not np.isfinite(scores).all()
                or not np.allclose(scores.sum(1),1,atol=1e-5,rtol=0)
                or [bool(v.argmax()>0 and v.max()>=.8) for v in scores]!=r['accepted']
                or [bool(v.argmax()==0) for v in scores]!=r['background_correct']
                or wider_crop(r['boxes'][0],(240,320,3))!=r['boxes'][1]):
            raise ValueError('Negative probe summary differs')
    accepted=[sum(r['accepted'][i] for r in doc['probes']) for i in (0,1)]
    gate=not effect['lost_targets'] and all(v['fp']<=0 for v in effect['count_delta'].values()) and accepted[1]<accepted[0]
    if gate!=doc['eligible_for_development'] or accepted!=doc['background_false_acceptances']:
        raise ValueError('Context gate differs')
    write_json(OUTPUT,dict(source_sha256=sha256(SOURCE),script_sha256=sha256(__file__),metrics=measured,
        lost_targets=len(effect['lost_targets']),gained_targets=len(effect['gained_targets']),
        background_false_acceptances=accepted,eligible_for_development=gate,
        decision='Inference-only context expansion rejected: fewer clutter false alarms but large TRAIN recall loss.',
        next_experiment='Matched context-distillation retraining: source pixels/crop policy must agree between training and inference. Do not force perfect fit of13ambiguous crops with a post-hoc head correction.',
        scope='TRAIN-only ablation, not held-out accuracy. Same classifier/proposals/call count; Pi speed unmeasured.',
        development_evaluated=False,test_evaluated=False,pi_tested=False,deployment_approved=False,qualification_achieved=False))
    print(dict(eligible_for_development=gate,lost_targets=len(effect['lost_targets']),background_false_acceptances=accepted),flush=True)


if __name__=='__main__':
    summarize()
