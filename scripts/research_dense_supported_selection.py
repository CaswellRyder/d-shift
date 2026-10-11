"""Research-only dense agreement for nested region selection, not new detections.

Fixed rule from TRAIN error inspection: a >=0.5-confidence same-color dense box
must overlap a raw-accepted child at IoU>=0.5 and its containing parent at IoU<0.25.
Suppress the parent before the existing selection policy, never change crop scores
or revive rejected classes. No thresholds selected on development images.
"""
from dtr.tracking import iou
from scripts.pi_balloon_search_bench import select_detections


def candidate_pairs(found):
    """Cheap necessary condition; no pair means no dense-model call is needed."""
    labels=('red_balloon','blue_balloon')
    pairs=[]
    for i,parent in enumerate(found):
        if not parent['accepted'] or parent['label'] not in labels:
            continue
        a,b,c,d=parent['box']
        area=(c-a)*(d-b)
        for j,child in enumerate(found):
            if i==j or not child['accepted'] or child['label']!=parent['label']:
                continue
            x,y,z,w=child['box']
            child_area=(z-x)*(w-y)
            overlap=max(0,min(c,z)-max(a,x))*max(0,min(d,w)-max(b,y))
            if child_area<=0 or child_area>=area or overlap<.9*child_area:
                continue
            pairs.append((i,j))
    return pairs


def supported_selection(found,learned):
    rows=[dict(r) for r in found]
    labels=('red_balloon','blue_balloon')
    rejected={}
    for i,j in candidate_pairs(found):
        if i not in rejected:
            parent,child=found[i],found[j]
            evidence=[r for r in learned if r['color_group']==labels.index(parent['label'])
                      and r['proposal_score']>=.5 and iou(r['box'],child['box'])>=.5
                      and iou(r['box'],parent['box'])<.25]
            if evidence:
                rejected[i]=j
    # Avoid contradictory chains of parent/child replacements.
    rejected={i:j for i,j in rejected.items() if j not in rejected}
    for i in rejected:
        rows[i]['accepted']=False
    selected=select_detections(rows,'split_merged',2.42913421076193)
    for i,j in rejected.items():
        selected[i].update(raw_accepted=True,suppressed=True,
            suppression_reason='dense_supported_child',dense_supported_child=j)
    return selected
