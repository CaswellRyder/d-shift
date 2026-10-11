"""Research removal of grid peak suppression; unchanged heat threshold and box NMS."""
import numpy as np

from dtr.tracking import iou
from scripts.research_blue_peak_blend import cell_box


def decode(values):
    if values.shape != (15, 20, 6) or not np.isfinite(values).all():
        raise ValueError('Require finite15x20x6dense output')
    heat = 1/(1+np.exp(-np.clip(values[..., :2], -80, 80)))
    pool = []
    for group in (0, 1):
        for y, x in np.argwhere(heat[..., group] >= .1):
            box = cell_box(values, y, x)
            if box is None:
                continue
            if group == 1:
                boxes, scores = [box], [float(heat[y, x, group])]
                for ny in range(max(0, y-1), min(15, y+2)):
                    for nx in range(max(0, x-1), min(20, x+2)):
                        if (ny == y and nx == x) or heat[ny, nx, group] < max(.1, .95*heat[y, x, group]):
                            continue
                        neighbor = cell_box(values, ny, nx)
                        if neighbor is not None and iou(box, neighbor) >= .5:
                            boxes.append(neighbor)
                            scores.append(float(heat[ny, nx, group]))
                if len(boxes) > 1:
                    box = np.average(np.asarray(boxes, np.float64), axis=0, weights=scores).tolist()
            a, b, c, d = box
            pad = max(c-a, d-b)*.12
            pool.append(dict(box=list(map(float, box)), crop_box=[max(0, int(np.floor(a-pad))),
                max(0, int(np.floor(b-pad))), min(320, int(np.ceil(c+pad))), min(240, int(np.ceil(d+pad)))],
                area=float((c-a)*(d-b)), color_group=group, proposal_score=float(heat[y, x, group])))
    found = []
    for row in sorted(pool, key=lambda r: -r['proposal_score']):
        if not any(r['color_group'] == row['color_group'] and iou(r['box'], row['box']) > .65 for r in found):
            found.append(row)
        if len(found) == 12:
            break
    return found
