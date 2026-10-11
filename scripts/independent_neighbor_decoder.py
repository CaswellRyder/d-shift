"""Bounded rescue of confident neighboring boxes describing distinct objects.

Keep the frozen blue-blended peaks. A non-peak cell can join only at the unchanged
direct acceptance threshold0.5 and only when no stronger same-color neighbor has
an overlapping box (IoU>=0.5). Also reject overlap>=0.5 with an existing peak.
Rank the combined pool at the existing12box ceiling; no ground truth is used.
"""
import cv2
import numpy as np

from dtr.tracking import iou
from scripts.research_blue_peak_blend import decode as frozen_decode, cell_box


def decode(values):
    base = frozen_decode(values, True)
    heat = 1/(1+np.exp(-np.clip(values[..., :2], -80, 80)))
    maxima = cv2.dilate(heat, np.ones((3, 3), np.uint8))
    pool = list(base)
    boxes = {}

    def box_at(y, x):
        if (y, x) not in boxes:
            boxes[y, x] = cell_box(values, y, x)
        return boxes[y, x]

    for group in (0, 1):
        for y, x in np.argwhere((heat[..., group] >= .5) & (heat[..., group] < maxima[..., group])):
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
            if duplicate:
                continue
            a, b, c, d = box
            pad = max(c-a, d-b)*.12
            pool.append(dict(box=list(map(float, box)), crop_box=[max(0, int(np.floor(a-pad))),
                max(0, int(np.floor(b-pad))), min(320, int(np.ceil(c+pad))), min(240, int(np.ceil(d+pad)))],
                area=float((c-a)*(d-b)), color_group=group, proposal_score=float(heat[y, x, group]),
                neighbor_rescue=True))
    if len(pool) == len(base):
        return base
    found = []
    for row in sorted(pool, key=lambda r: -r['proposal_score']):
        if not any(r['color_group'] == row['color_group'] and iou(r['box'], row['box']) > .65 for r in found):
            found.append(row)
        if len(found) == 12:
            break
    return found
