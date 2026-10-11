"""Bounded luminance-edge ellipse proposals; research only, no semantic bypass.

Unlike the earlier color-mask radial fit, these boundaries come from luminance
edges, which can exist between touching objects of similar blue chromaticity.
All constants are fixed before TRAIN coverage and development evaluation.
"""
import cv2
import numpy as np

from dtr.tracking import iou
from scripts.research_balloon_search import experimental,select


def fit_candidate(points, blue):
    if len(points)<16:
        return None
    try:
        (cx,cy),(diameter_a,diameter_b),degrees=cv2.fitEllipse(points.astype(np.float32))
    except cv2.error:
        return None
    if (not np.isfinite([cx,cy,diameter_a,diameter_b,degrees]).all()
            or not 0<=cx<320 or not 0<=cy<240
            or min(diameter_a,diameter_b)<8 or max(diameter_a,diameter_b)>200
            or max(diameter_a,diameter_b)/min(diameter_a,diameter_b)>2.8):
        return None
    a,b=diameter_a/2,diameter_b/2
    theta=np.deg2rad(degrees)
    rotation=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    normalized=(points.reshape(-1,2)-[cx,cy])@rotation/[a,b]
    residual=np.abs(np.linalg.norm(normalized,axis=1)-1)
    angles=np.mod(np.arctan2(normalized[:,1],normalized[:,0]),2*np.pi)
    sectors=len(np.unique(np.floor(angles*12/(2*np.pi)).astype(int)))
    if sectors<6 or np.median(residual)>.10 or np.quantile(residual,.9)>.25:
        return None
    ring=np.arange(16)*2*np.pi/16
    unit=np.column_stack((np.cos(ring),np.sin(ring)))

    def fraction(scale):
        sample=np.rint((unit*[a,b]*scale)@rotation.T+[cx,cy]).astype(int)
        visible=(sample[:,0]>=0)&(sample[:,0]<320)&(sample[:,1]>=0)&(sample[:,1]<240)
        if visible.sum()<8:
            return None
        return float(blue[sample[visible,1],sample[visible,0]].mean())

    inner,outer=fraction(.6),fraction(1.2)
    if inner is None or outer is None or inner<.6 or inner-outer<.2:
        return None
    extent=np.sqrt((rotation*rotation)@np.array([a*a,b*b]))
    low=np.floor([cx,cy]-extent).astype(int)
    high=np.ceil([cx,cy]+extent).astype(int)+1
    x,y=max(0,int(low[0])),max(0,int(low[1]))
    z,w=min(320,int(high[0])),min(240,int(high[1]))
    if min(z-x,w-y)<4:
        return None
    pad=int(max(z-x,w-y)*.12)
    area=float(np.pi*a*b)
    return dict(box=[x,y,z,w],crop_box=[max(0,x-pad),max(0,y-pad),min(320,z+pad),min(240,w+pad)],
                color_group=1,area=area,proposal_score=float(np.sqrt(area)*sectors/12*(inner-outer)),
                proposal_origin='luminance_edge_ellipse',edge_sectors=sectors,
                median_edge_residual=float(np.median(residual)),inner_blue_fraction=inner,
                outer_blue_fraction=outer)


def edge_candidates(rgb):
    if rgb.dtype!=np.uint8 or rgb.shape!=(240,320,3):
        raise ValueError('Require 320x240 RGB uint8')
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    smooth=cv2.GaussianBlur(gray,(3,3),.8)
    edges=cv2.Canny(smooth,40,100)
    contours,_=cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    blue=cv2.subtract(rgb[:,:,2],cv2.max(rgb[:,:,0],rgb[:,:,1]))>=15
    pool=[]
    # At most 24 contours, 13 bounded fits each, 256 points per fit.
    for contour in sorted(contours,key=len,reverse=True)[:24]:
        if len(contour)<16:
            continue
        points=contour[::max(1,int(np.ceil(len(contour)/256)))]
        hypotheses=[points]
        length=len(points)
        if length>=32:
            for start in (0,length//4,length//2,3*length//4):
                hypotheses.append(points[(np.arange(length//2)+start)%length])
        if length>=64:
            for step in range(8):
                hypotheses.append(points[(np.arange(length//4)+step*length//8)%length])
        for hypothesis in hypotheses:
            row=fit_candidate(hypothesis,blue)
            if row:
                pool.append(row)
    return select(pool,limit=8)


def proposals(rgb):
    baseline=experimental(rgb,'mser')
    extras=[]
    for row in edge_candidates(rgb):
        if not any(iou(row['box'],old['box'])>.65 for old in baseline+extras):
            extras.append(row)
        if len(extras)==2 or len(baseline)+len(extras)>=12:
            break
    # Keep every baseline slot even if a future baseline uses all twelve.
    return baseline+extras[:max(0,12-len(baseline))]
