"""Research candidate allocation by color and scale, within the unchanged12-call ceiling."""
import cv2
import numpy as np

from dtr.tracking import iou
from scripts.research_balloon_search import candidate, select
from scripts.research_blue_bright_search import impossible_geometry


def balanced_select(pool):
    """Up to3 slots per color/scale; unused slots fall back to original ranking."""
    ordered = sorted(pool, key=lambda c: -c['proposal_score'])
    queues = []
    for small in (True, False):
        for group in (0, 1):
            queues.append([c for c in ordered if c['color_group'] == group and
                (max(c['box'][2] - c['box'][0], c['box'][3] - c['box'][1]) <= 32) == small])
    found = []
    for _ in range(3):
        for queue in queues:
            while queue:
                c = queue.pop(0)
                if not any(iou(c['box'], p['box']) > .65 for p in found):
                    found.append(c)
                    break
    # Use the old color-alternating score ranking for remaining capacity.
    for c in select(pool, limit=len(pool)):
        if len(found) == 12:
            break
        if not any(iou(c['box'], p['box']) > .65 for p in found):
            found.append(c)
    return found


def pool(rgb, native):
    if rgb.shape != (240, 320, 3) or rgb.dtype != np.uint8:
        raise ValueError('Require320x240RGB uint8')
    red, green, blue = cv2.split(rgb)
    found = []
    for group, plane in enumerate((cv2.subtract(red, cv2.max(green, blue)),
                                    cv2.subtract(blue, cv2.max(red, green)))):
        detector = cv2.MSER_create(5, 6, 25000)
        if group == 1:
            detector.setPass2Only(True)
        regions, bounds = detector.detectRegions(plane)
        for points, (_, _, width, height) in zip(regions, bounds, strict=True):
            if impossible_geometry(int(width), int(height)):
                continue
            reduced = native.reduce(plane, points)
            if len(reduced):
                c = candidate(reduced, group, rgb.shape)
                if c:
                    found.append(c)
    return found
