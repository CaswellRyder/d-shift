"""Attribute frozen development false alarms to box/crop geometry; no mining."""
from collections import Counter
from pathlib import Path

from dtr.data import read_json,sha256,write_json
from scripts.evaluate_runtime_positive_weight16 import RAW,OUTPUT as DEVELOPMENT
from scripts.summarize_public_balloon_trial import checked_metrics
from scripts.audit_balloon_false_positives import classify_errors

OUTPUT = Path('research/balloon-red-blue-20261008/crop-box-mismatch-20261011.json')


def intersection(a,b):
    return max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))


def crop_geometry(detection,truth):
    same = [t['box'] for t in truth if t['label']==detection['label']]
    other = [t['box'] for t in truth if t['label']!=detection['label']]
    crop,box = detection['crop_box'],detection['box']
    area = (crop[2]-crop[0])*(crop[3]-crop[1])
    if area<=0:
        raise ValueError('Invalid crop')
    crop_intersections = [intersection(crop,t) for t in same]
    return dict(same_color_in_crop=any(v>0 for v in crop_intersections),
        same_color_in_box=any(intersection(box,t)>0 for t in same),
        max_same_color_crop_fraction=max((v/area for v in crop_intersections),default=0.),
        max_same_color_target_fraction=max((v/((t[2]-t[0])*(t[3]-t[1])) for v,t in zip(crop_intersections,same,strict=True)),default=0.),
        other_color_in_crop=any(intersection(crop,t)>0 for t in other))


def audit():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    doc,raw = read_json(DEVELOPMENT),read_json(RAW)
    if doc['raw']['sha256']!=sha256(RAW):
        raise ValueError('Source changed')
    panels = {}
    for panel,variants in raw.items():
        result = variants['parent/coarse47']
        if checked_metrics(result)!=doc['panels'][panel]['results']['parent/coarse47']:
            raise ValueError('Metrics changed')
        rows,counts = [],Counter()
        for frame,baseline in zip(result['frames'],variants['parent/baseline']['frames'],strict=True):
            if any(frame[k]!=baseline[k] for k in ('source','truth','frame_sha256')):
                raise ValueError('Different scenes')
            for e in classify_errors(frame,baseline):
                candidates = [r for r in frame['detections'] if r['accepted'] and r['box']==e['box']
                    and r['label']==e['label'] and r['score']==e['score']]
                if len(candidates)!=1:
                    raise ValueError('Ambiguous detection identity')
                r = candidates[0]
                geometry = crop_geometry(r,frame['truth'])
                category = ('same_color_in_classifier_crop' if geometry['same_color_in_crop']
                            else 'other_color_only_in_crop' if geometry['other_color_in_crop'] else 'no_annotated_balloon_in_crop')
                counts[e['label']+'/'+category] += 1
                rows.append(dict(e,source=frame['source'],crop_box=r['crop_box'],geometry=geometry))
        panels[panel] = dict(counts=dict(counts),errors=rows)
    write_json(OUTPUT,dict(panels=panels,source_receipt_sha256=sha256(DEVELOPMENT),raw_sha256=sha256(RAW),
        script_sha256=sha256(__file__),error_helper_sha256=sha256('scripts/audit_balloon_false_positives.py'),
        scope='Geometric attribution on previously used development only. Annotation intersection is not semantic proof. No training pixels, labels, thresholds or runtime changes.',
        test_evaluated=False,deployment_approved=False,qualification_achieved=False))
    print({p:r['counts'] for p,r in panels.items()})


if __name__=='__main__':
    audit()
