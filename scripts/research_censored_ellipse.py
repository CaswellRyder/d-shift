"""Research-only free-center ellipse fit with image-edge rays censored.

No labels, confidence thresholds, model changes or runtime defaults. At most
12 core fits, four 32-point ellipse fits per core, two extra classifier calls.
"""
import cv2
import numpy as np

from dtr.tracking import iou
from scripts.research_balloon_search import blue_core_candidates, core_rays, experimental, select


def fit_box(mask, x, y, radius):
    directions, lengths = core_rays(mask, x, y, radius)
    points = np.array([x, y]) + directions * lengths[:, None]
    # A stop at the image border or ray-length cap is not a measured boundary.
    visible = ((points[:, 0] > 1) & (points[:, 0] < 318)
               & (points[:, 1] > 1) & (points[:, 1] < 238)
               & (lengths < np.ceil(3 * radius) - 1) & (lengths >= 2))
    points = points[visible]
    if len(points) < 16:
        return None
    normalized = (points - [x, y]) / radius
    keep = np.arange(len(points))
    for _ in range(4):
        try:
            center, diameters, angle = cv2.fitEllipse(normalized[keep].astype(np.float32))
        except cv2.error:
            return None
        if not np.isfinite([*center, *diameters, angle]).all() or min(diameters) <= 0:
            return None
        theta = np.deg2rad(angle)
        rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        axes = np.array(diameters) / 2
        residual = np.abs(np.linalg.norm((normalized - center) @ rotation / axes, axis=1) - 1)
        keep = np.argsort(residual, kind='stable')[:max(16, int(np.ceil(.75 * len(points))))]
    if (min(axes) < .5 or max(axes) > 3 or max(axes) / min(axes) > 3
            or np.linalg.norm(center) > 1 or np.median(residual) > .15):
        return None
    center = np.array([x, y]) + np.array(center) * radius
    extents = radius * np.sqrt((rotation * rotation) @ (axes * axes))
    low, high = np.floor(center - extents).astype(int), np.ceil(center + extents).astype(int) + 1
    box = [max(0, int(low[0])), max(0, int(low[1])), min(320, int(high[0])), min(240, int(high[1]))]
    if min(box[2] - box[0], box[3] - box[1]) < 4:
        return None
    return box


def candidates(rgb):
    if rgb.shape != (240, 320, 3) or rgb.dtype != np.uint8:
        raise ValueError('Require320x240 RGB uint8')
    plane = cv2.subtract(rgb[:, :, 2], cv2.max(rgb[:, :, 0], rgb[:, :, 1]))
    contours, _ = cv2.findContours(cv2.compare(plane, 15, cv2.CMP_GE), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros((242, 322), np.uint8)
    for contour in contours:
        cv2.drawContours(mask, [contour + 1], -1, 255, -1)
    pool = []
    for core in blue_core_candidates(rgb)[:12]:
        box = fit_box(mask, *core['core_center'], core['core_radius'])
        if box is None:
            continue
        a, b, c, d = box
        area = float(np.count_nonzero(mask[b+1:d+1, a+1:c+1]))
        pad = int(max(c-a, d-b) * .12)
        pool.append(dict(core, box=box,
            crop_box=[max(0, a-pad), max(0, b-pad), min(320, c+pad), min(240, d+pad)],
            area=area, proposal_score=float(np.sqrt(area) * (area / ((c-a)*(d-b)))**2),
            proposal_source='blue_censored_free_ellipse'))
    return select(pool)


def proposals(rgb):
    base = experimental(rgb, 'mser')
    extras = []
    for row in candidates(rgb):
        if len(extras) >= 2 or len(base) + len(extras) >= 12:
            break
        if not any(iou(row['box'], old['box']) > .65 for old in base + extras):
            extras.append(row)
    return base + extras
