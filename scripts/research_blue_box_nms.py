"""TRAIN-gated blue non-peak hypotheses, retaining the existing crop budget."""
import cv2
import numpy as np

from dtr.tracking import iou
from scripts.research_blue_peak_blend import cell_box, decode as parent_decode


def decode(values):
    pool = parent_decode(values, True)  # Validates shape/finite values; no mutation.
    heat = 1/(1+np.exp(-np.clip(values[..., 1], -80, 80)))
    peaks = heat == cv2.dilate(heat, np.ones((3, 3), np.uint8))
    for y, x in np.argwhere((heat >= .1) & ~peaks):
        box = cell_box(values, y, x)
        if box is None:
            continue
        a, b, c, d = box
        pad = max(c-a, d-b)*.12
        pool.append(dict(box=[float(v) for v in box],
            crop_box=[max(0, int(np.floor(a-pad))), max(0, int(np.floor(b-pad))),
                      min(320, int(np.ceil(c+pad))), min(240, int(np.ceil(d+pad)))],
            area=float((c-a)*(d-b)), color_group=1, proposal_score=float(heat[y, x])))
    found = []
    for row in sorted(pool, key=lambda r: -r['proposal_score']):
        if not any(r['color_group'] == row['color_group'] and iou(r['box'], row['box']) > .65 for r in found):
            found.append(row)
        if len(found) == 12:
            break
    return found
