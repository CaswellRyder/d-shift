"""Shrink accepted blue boxes only; preserve original classification evidence."""
import math

from dtr.tracking import iou


def geometry(box):
    if len(box)!=4 or not all(math.isfinite(v) for v in box):
        raise ValueError('Invalid box')
    a,b,c,d=box
    if not (0<=a<c<=320 and 0<=b<d<=240):
        raise ValueError('Invalid box bounds')
    return c-a,d-b,(a+c)/2,(b+d)/2,(c-a)*(d-b)


def eligible(parent, proposal):
    if not math.isfinite(proposal['proposal_score']) or not 0<=proposal['proposal_score']<=1:
        raise ValueError('Invalid proposal score')
    if proposal['color_group']!=1 or proposal['proposal_score']<.5:
        return False
    w,h,cx,cy,area=geometry(parent)
    _,_,nx,ny,new_area=geometry(proposal['box'])
    a,b,c,d=parent
    e,f,g,k=proposal['box']
    intersection=max(0,min(c,g)-max(a,e))*max(0,min(d,k)-max(b,f))
    return (.35<=new_area/area<=.8 and intersection/new_area>=.9
            and abs(nx-cx)<=.25*w and abs(ny-cy)<=.25*h)


def refine(detections, proposals):
    rows=[dict(r) for r in detections]
    # Work on accepted rows only. Never accept a rejected classifier result.
    candidates=sorted((i for i,r in enumerate(rows) if r['accepted'] and r['label']=='blue_balloon'),
                      key=lambda i:-rows[i]['score'])
    changed=0
    used=set()
    for i in candidates:
        choices=[]
        for j,p in enumerate(proposals):
            if j in used or not eligible(rows[i]['box'],p):
                continue
            if any(k!=i and r['accepted'] and iou(r['box'],p['box'])>.65 for k,r in enumerate(rows)):
                continue
            choices.append(j)
        if not choices:
            continue
        j=max(choices,key=lambda j:proposals[j]['proposal_score'])
        proposal=proposals[j]
        rows[i].update(pre_refinement_box=list(rows[i]['box']),box=list(proposal['box']),
                       area=geometry(proposal['box'])[-1],refinement_score=proposal['proposal_score'],
                       refinement_reason='contained_blue_geometry')
        used.add(j)
        changed+=1
        if changed==2:
            break
    return rows
