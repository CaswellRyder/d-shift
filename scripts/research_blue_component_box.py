"""Research-only geometry correction for learned blue proposals; never a color veto.

Use the largest untruncated blue-opponent component only when it owns at least
half the mask support inside the proposal. Otherwise retain the original box.
Keep classification crop, scores and acceptance unchanged. No added neural calls.
"""
import math

import cv2
import numpy as np


def refine(rgb, found, base_count):
    if rgb.shape != (240, 320, 3) or rgb.dtype != np.uint8:
        raise ValueError('Require320x240 RGB uint8')
    if type(base_count) is not int or not 0 <= base_count <= len(found) <= 12:
        raise ValueError('Invalid baseline prefix')
    rows = [dict(row) for row in found]
    plane = cv2.subtract(rgb[:, :, 2], cv2.max(rgb[:, :, 0], rgb[:, :, 1]))
    for i, row in enumerate(rows):
        if i < base_count or row['label'] != 'blue_balloon' or not row['accepted']:
            continue
        box = row['box']
        if (len(box) != 4 or not all(math.isfinite(v) for v in box)
                or not 0 <= box[0] < box[2] <= 320 or not 0 <= box[1] < box[3] <= 240):
            raise ValueError('Invalid proposal geometry')
        a, b = math.floor(box[0]), math.floor(box[1])
        c, d = math.ceil(box[2]), math.ceil(box[3])
        mask = (plane[b:d, a:c] >= 15).astype(np.uint8)
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        row.update(original_box=list(box), box_refined=False)
        if len(stats) < 2:
            row['box_refinement_reason'] = 'no_blue_support'
            continue
        x, y, w, h, count = map(int, stats[1:][np.argmax(stats[1:, cv2.CC_STAT_AREA])])
        if count < 6 or min(w, h) < 4 or count < .5*int(mask.sum()):
            row['box_refinement_reason'] = 'weak_or_fragmented_support'
            continue
        if x == 0 or y == 0 or x+w == c-a or y+h == d-b:
            row['box_refinement_reason'] = 'support_touches_roi_boundary'
            continue
        row.update(box=[a+x, b+y, a+x+w, b+y+h], box_refined=True,
            box_refinement_reason='dominant_complete_blue_component')
    return rows
