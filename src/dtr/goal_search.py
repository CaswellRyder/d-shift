"""Opt-in, color-independent acquisition proposals; not a default or a detector.

At most four edge proposals share the EXISTING candidate budget. No target labels
are inferred from edges. Fixed thresholds are a hypothesis for development tests.
"""
import cv2
import numpy as np

from .tracking import iou


def edge_candidates(rgb, limit=4):
    if type(limit) is not int or not 0 <= limit <= 4:
        raise ValueError("Edge budget must be 0..4")
    if (rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3
            or min(rgb.shape[:2]) < 4 or rgb.shape[0] > 240 or rgb.shape[1] > 320):
        raise ValueError("Edge search requires uint8 RGB, at most 320x240")
    if limit == 0:
        return []
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 30, 90)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    height, width = gray.shape
    candidates = []
    for contour in contours:
        if len(contour) < 3:
            continue
        area = cv2.contourArea(contour)
        if area < 6:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if min(w, h) < 4 or max(w, h) > 96 or max(w/h, h/w) > 4:
            continue
        perimeter = cv2.arcLength(contour, True)
        compactness = 4*np.pi*area / max(perimeter*perimeter, 1)
        if compactness < .15:
            continue
        patch = gray[y:y+h, x:x+w]
        contrast = int(patch.max()) - int(patch.min())
        pad = int(max(w, h)*.12)
        candidates.append(dict(
            box=[x, y, x+w, y+h],
            crop_box=[max(0, x-pad), max(0, y-pad), min(width, x+w+pad), min(height, y+h+pad)],
            area=float(area), color_group=-1, proposal_source="grayscale_edges",
            proposal_score=float(compactness*contrast)))
    selected = []
    for candidate in sorted(candidates, key=lambda r: (-r["proposal_score"], r["box"])):
        if any(iou(candidate["box"], other["box"]) > .5 for other in selected):
            continue
        selected.append(candidate)
        if len(selected) == limit:
            break
    return selected


def mixed_candidates(rgb, baseline, limit):
    """Retain first color candidates; reserve <=one third of slots for new edges."""
    reserve = min(4, limit//3)
    # Filter duplicates against the full baseline before replacing tail entries.
    edges = [r for r in edge_candidates(rgb, reserve)
             if not any(iou(r["box"], c["box"]) > .5 for c in baseline)]
    return baseline[:limit-len(edges)] + edges
