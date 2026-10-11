"""Research-only exact-semantics decoder optimization; no threshold changes.

Share sigmoid, dilation and per-cell geometry between peak and neighbor passes.
Convert decoded NumPy scalars to Python doubles before the small scalar IoU loops.
The reference cell arithmetic and blue weighted averaging remain unchanged.
"""
import cv2
import numpy as np

from dtr.tracking import iou
from scripts.research_blue_peak_blend import cell_box


def proposal(box, score, group, rescue=False):
    a, b, c, d = box
    pad = max(c-a, d-b)*.12
    row = dict(box=list(box), crop_box=[max(0, int(np.floor(a-pad))),
        max(0, int(np.floor(b-pad))), min(320, int(np.ceil(c+pad))), min(240, int(np.ceil(d+pad)))],
        area=float((c-a)*(d-b)), color_group=group, proposal_score=float(score))
    if rescue:
        row['neighbor_rescue'] = True
    return row


def suppress(pool):
    found = []
    for row in sorted(pool, key=lambda r: -r['proposal_score']):
        if not any(r['color_group'] == row['color_group'] and iou(r['box'], row['box']) > .65 for r in found):
            found.append(row)
        if len(found) == 12:
            break
    return found


def decode(values, rescue=False):
    if values.shape != (15, 20, 6) or not np.isfinite(values).all():
        raise ValueError('Require finite dense proposal output')
    heat = 1/(1+np.exp(-np.clip(values[..., :2], -80, 80)))
    maxima = cv2.dilate(heat, np.ones((3, 3), np.uint8))
    boxes = {}

    def box_at(y, x):
        key = (y, x)
        if key not in boxes:
            box = cell_box(values, y, x)
            boxes[key] = None if box is None else list(map(float, box))
        return boxes[key]

    pool = []
    for group in (0, 1):
        peaks = (heat[..., group] >= .1) & (heat[..., group] == maxima[..., group])
        for y, x in np.argwhere(peaks):
            box = box_at(y, x)
            if box is None:
                continue
            if group == 1:
                blend_boxes, scores = [box], [float(heat[y, x, group])]
                for ny in range(max(0, y-1), min(15, y+2)):
                    for nx in range(max(0, x-1), min(20, x+2)):
                        if (ny == y and nx == x) or heat[ny, nx, group] < max(.1, .95*heat[y, x, group]):
                            continue
                        neighbor = box_at(ny, nx)
                        if neighbor is not None and iou(box, neighbor) >= .5:
                            blend_boxes.append(neighbor)
                            scores.append(float(heat[ny, nx, group]))
                if len(blend_boxes) > 1:
                    box = np.average(np.asarray(blend_boxes, np.float64), axis=0, weights=scores).tolist()
            pool.append(proposal(box, heat[y, x, group], group))
    base = suppress(pool)
    if not rescue:
        return base
    pool = list(base)
    for group in (0, 1):
        candidates = (heat[..., group] >= .5) & (heat[..., group] < maxima[..., group])
        for y, x in np.argwhere(candidates):
            box = box_at(y, x)
            if box is None or any(r['color_group'] == group and iou(box, r['box']) >= .5 for r in base):
                continue
            duplicate = False
            for ny in range(max(0, y-1), min(15, y+2)):
                for nx in range(max(0, x-1), min(20, x+2)):
                    if heat[ny, nx, group] <= heat[y, x, group]:
                        continue
                    neighbor = box_at(ny, nx)
                    if neighbor is not None and iou(box, neighbor) >= .5:
                        duplicate = True
            if not duplicate:
                pool.append(proposal(box, heat[y, x, group], group, True))
    return base if len(pool) == len(base) else suppress(pool)
