"""Bounded Lab-color splitting of blue regions, research only.

Three deterministic color clusters can separate a cyan balloon from darker
blue clutter joined in opponent-color space. No truth, learned thresholds,
camera, serial, deployment defaults or neural weights enter proposal creation.
"""
import argparse
from pathlib import Path
import time

import cv2
import numpy as np

from dtr.data import sha256, write_json
from dtr.tracking import iou
from scripts.research_balloon_search import candidate, experimental, select


def centers_for(points):
    """Deterministic farthest-point initialization, then five Lloyd steps."""
    points=np.asarray(points,dtype=np.float32)
    first=int(np.argmin(np.sum((points-points.mean(axis=0))**2,axis=1)))
    centers=[points[first]]
    for _ in range(2):
        distance=np.min(np.sum((points[:,None,:]-np.asarray(centers)[None,:,:])**2,axis=2),axis=1)
        if float(distance.max()) < 1:
            return None
        centers.append(points[int(distance.argmax())])
    centers=np.asarray(centers)
    for _ in range(5):
        labels=np.sum((points[:,None,:]-centers[None,:,:])**2,axis=2).argmin(axis=1)
        centers=np.asarray([points[labels==k].mean(axis=0) if np.any(labels==k) else centers[k]
                            for k in range(3)],dtype=np.float32)
    return centers


def color_split_candidates(rgb):
    if rgb.dtype != np.uint8 or rgb.shape != (240,320,3):
        raise ValueError("Require 320x240 RGB uint8")
    plane=cv2.subtract(rgb[:,:,2],cv2.max(rgb[:,:,0],rgb[:,:,1]))
    ys,xs=np.nonzero(plane>=15)
    if len(xs)<48:
        return []
    lab=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB)
    values=lab[ys,xs].astype(np.float32)
    stride=max(1,int(np.ceil(len(values)/16000)))
    centers=centers_for(values[::stride])
    if centers is None:
        return []
    labels=np.sum((values[:,None,:]-centers[None,:,:])**2,axis=2).argmin(axis=1)
    pool=[]
    # Union hypotheses retain objects whose highlights/shadows span clusters.
    # All returned boxes still require normal classifier and selection stages.
    for group in ((0,),(1,),(2,),(0,1),(0,2),(1,2)):
        keep=np.isin(labels,group)
        mask=np.zeros((240,320),np.uint8)
        mask[ys[keep],xs[keep]]=255
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for contour in sorted(contours,key=cv2.contourArea,reverse=True)[:16]:
            c=candidate(contour,1,rgb.shape)
            if c:
                pool.append(dict(c,proposal_origin="lab_blue_split",color_clusters=list(group)))
    return select(pool,limit=12)


def tiny_nested(row, parents):
    """Research guard: reject a Lab fragment under 20% of an enclosing blue box.

    Fixed geometric hypothesis, not a learned or deployment-approved threshold.
    A real small balloon against larger blue clutter may also be lost; audit the
    one-to-one TRAIN proposal ceiling before checking development scenes.
    """
    a,b,c,d=row["box"]
    area=max(0,c-a)*max(0,d-b)
    if area <= 0:
        return True
    for parent in parents:
        if parent["color_group"] != row["color_group"]:
            continue
        x,y,z,w=parent["box"]
        overlap=max(0,min(c,z)-max(a,x))*max(0,min(d,w)-max(b,y))
        if area < .2*max(0,z-x)*max(0,w-y) and overlap >= .9*area:
            return True
    return False


def proposals(rgb, fragment_guard=False):
    baseline=experimental(rgb,"mser")
    extras=[]
    for row in color_split_candidates(rgb):
        if fragment_guard and tiny_nested(row,baseline):
            continue
        if not any(iou(row["box"],old["box"])>.65 for old in baseline+extras):
            extras.append(row)
        if len(extras)==4:
            break
    return baseline+extras


def training_audit(original_manifest, original_archive, indoor_archive, queues):
    from scripts.audit_balloon_search_stages import original_frames, indoor_frames
    from scripts.audit_balloon_pipeline_errors import maximum_matching
    panels={"original":original_frames(original_manifest,original_archive)}
    for queue,review in queues:
        panels[str(queue)]=indoor_frames(queue,review,original_manifest,indoor_archive)
    results={}
    for name,frames in panels.items():
        outputs=[]
        for source,rgb,truth in frames:
            base=experimental(rgb,"mser")
            start=time.perf_counter()
            combined=proposals(rgb)
            elapsed=(time.perf_counter()-start)*1000
            guarded=proposals(rgb,fragment_guard=True)
            if combined[:len(base)]!=base:
                raise ValueError("Baseline prefix changed")
            scores={}
            for label,group in (("red_balloon",0),("blue_balloon",1)):
                selected=[r for r in truth if r["label"]==label]
                scores[label]=dict(targets=len(selected),**{
                    variant:len(maximum_matching(selected,rows,lambda t,d:d["color_group"]==group))
                    for variant,rows in (("baseline",base),("color_split",combined),("fragment_guard",guarded))})
            outputs.append(dict(source=source,coverage=scores,extra_calls=len(combined)-len(base),
                                guarded_extra_calls=len(guarded)-len(base),
                                host_search_ms=elapsed))
        totals={label:{k:sum(f["coverage"][label][k] for f in outputs)
                       for k in ("targets","baseline","color_split","fragment_guard")}
                for label in ("red_balloon","blue_balloon")}
        results[name]=dict(frames=len(outputs),coverage=totals,details=outputs)
    return dict(scope="Training-only one-to-one proposal coverage, not detector precision/recall or Pi timing",
                original_manifest_sha256=sha256(original_manifest),original_archive_sha256=sha256(original_archive),
                indoor_archive_sha256=sha256(indoor_archive),
                queue_receipts=[dict(queue=str(q),queue_sha256=sha256(Path(q)/"review.json"),
                                     review_sha256=sha256(r)) for q,r in queues],
                script_sha256=sha256(__file__),baseline_search_sha256=sha256("scripts/research_balloon_search.py"),
                results=results,test_evaluated=False,deployment_approved=False)


def evaluate(queue,review_path,model):
    from scripts.evaluate_reviewed_balloon_scenes import reviewed_frames
    from scripts.research_predictor import load_predictor, classify_candidates
    from scripts.pi_balloon_search_bench import select_detections
    from scripts.evaluate_red_blue_development import detection_counts,metrics,LABELS
    from dtr.vision import RED_BLUE_PROFILE,validate_model_profile
    frames,review=reviewed_frames(queue,review_path)
    predictor=load_predictor(model)
    validate_model_profile(predictor.metadata,RED_BLUE_PROFILE)
    if predictor.metadata["threshold"]!=.8 or review["threshold"]!=.8 or review["matching_iou"]!=.5:
        raise ValueError("Frozen operating point changed")
    variants={}
    for name,search in (("baseline",lambda rgb:experimental(rgb,"mser")),("color_split",proposals),
                        ("fragment_guard",lambda rgb:proposals(rgb,fragment_guard=True))):
        total={label:dict(tp=0,fp=0,fn=0) for label in LABELS}
        outputs=[]
        for row,rgb in frames:
            found=classify_candidates(predictor,rgb,search(rgb))
            detections=select_detections(found,"split_merged",2.42913421076193)
            counts=detection_counts(row["provisional_truth"],detections)
            for label in LABELS:
                for k in total[label]:
                    total[label][k]+=counts[label][k]
            outputs.append(dict(id=row["id"],source=row["source"],frame_sha256=row["sha256"],
                                truth=row["provisional_truth"],counts=counts,detections=detections))
        variants[name]=dict(metrics=metrics(total),frames=outputs)
    return dict(scope="Previously inspected development scenes, not independent qualification",variants=variants,
                queue_sha256=review["queue_sha256"],review_sha256=sha256(review_path),model_sha256=sha256(model),
                script_sha256=sha256(__file__),baseline_search_sha256=sha256("scripts/research_balloon_search.py"),
                selection="split_merged",split_max_aspect=2.42913421076193,threshold=.8,matching_iou=.5,
                test_evaluated=False,deployment_approved=False,pi_timing_measured=False)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode",choices=("train-audit","development"),required=True)
    p.add_argument("--original-manifest")
    p.add_argument("--original-archive")
    p.add_argument("--indoor-archive")
    p.add_argument("--queue",nargs=2,action="append",metavar=("ROOT","REVIEW"),required=True)
    p.add_argument("--model")
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    if a.mode=="train-audit":
        if not all((a.original_manifest,a.original_archive,a.indoor_archive)):
            p.error("Training audit requires original and indoor sources")
        result=training_audit(a.original_manifest,a.original_archive,a.indoor_archive,a.queue)
        print({k:v["coverage"] for k,v in result["results"].items()})
    else:
        if len(a.queue)!=1 or not a.model:
            p.error("Development requires exactly one queue and a model")
        result=evaluate(*a.queue[0],a.model)
        print({k:v["metrics"] for k,v in result["variants"].items()})
    write_json(a.output,result)
