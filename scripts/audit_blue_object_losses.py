"""Explain paired development losses without fitting or changing thresholds."""
from collections import Counter
from pathlib import Path

from dtr.data import read_json,sha256,write_json
from dtr.tracking import iou
from scripts.evaluate_blue_objects import OUTPUT as DEVELOPMENT, RAW
from scripts.audit_current_blue_search import matched_truth
from scripts.audit_balloon_false_positives import candidate_key
from scripts.summarize_public_balloon_trial import checked_metrics

OUTPUT=Path('research/balloon-red-blue-20261008/blue-object-losses-20261011.json')


def failure_reason(target,rows):
    localized=[r for r in rows if iou(target['box'],r['box'])>=.5]
    if not localized:
        return 'no_localized_proposal'
    same=[r for r in localized if r['label']==target['label']]
    if not same:
        return 'class_changed'
    if not any(r['raw_accepted'] for r in same):
        return 'confidence_below_threshold'
    if not any(r['accepted'] for r in same):
        return 'selection_suppressed'
    return 'matching_assignment'


def audit():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    doc,raw=read_json(DEVELOPMENT),read_json(RAW)
    if doc['raw']['sha256']!=sha256(RAW):
        raise ValueError('Evaluation changed')
    results=[]
    for panel,variants in raw.items():
        for key,result in variants.items():
            if checked_metrics(result)!=doc['panels'][panel]['results'][key]:
                raise ValueError('Metrics changed')
            if not key.startswith('admit'):
                continue
            base=variants['parent/'+key.split('/')[1]]
            counts,losses=Counter(),[]
            for f,old in zip(result['frames'],base['frames'],strict=True):
                if any(f[k]!=old[k] for k in ('source','truth','frame_sha256')):
                    raise ValueError('Different frames')
                old_rows={candidate_key(r):r for r in old['detections']}
                if (len(old_rows)!=len(old['detections']) or len(f['detections'])!=len(old_rows)
                        or {candidate_key(r) for r in f['detections']}!=set(old_rows)):
                    raise ValueError('Proposal geometry changed')
                for i in sorted(matched_truth(old)-matched_truth(f)):
                    target=f['truth'][i]
                    reason=failure_reason(target,f['detections'])
                    counts[target['label']+'/'+reason]+=1
                    candidates=[]
                    for row in f['detections']:
                        if iou(target['box'],row['box'])<.5:
                            continue
                        previous=old_rows[candidate_key(row)]
                        fields=('label','score','scores','raw_accepted','accepted','suppressed')
                        candidates.append(dict(box=row['box'],crop_box=row['crop_box'],
                            before={k:previous[k] for k in fields},after={k:row[k] for k in fields}))
                    losses.append(dict(source=f['source'],target=target,reason=reason,localized_candidates=candidates))
            results.append(dict(panel=panel,variant=key,counts=dict(counts),losses=losses))
    write_json(OUTPUT,dict(results=results,development_sha256=sha256(DEVELOPMENT),raw_sha256=sha256(RAW),
        script_sha256=sha256(__file__),scope='Attribution of previously evaluated development losses, not calibration or TRAIN mining.',
        test_evaluated=False,pi_tested=False,deployment_approved=False,qualification_achieved=False))
    print([{k:r[k] for k in ('panel','variant','counts')} for r in results])


if __name__=='__main__':
    audit()
