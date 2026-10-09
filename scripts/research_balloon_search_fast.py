"""Exact MSER geometry optimization experiment; unchanged research semantics."""

import cv2
import numpy as np

try:
    from research_balloon_search import candidate, select
except ModuleNotFoundError:
    from scripts.research_balloon_search import candidate, select


def search_fast(rgb, native, bright_only=False):
    # Opt-in policy ablation: unlike direct pointers, this CAN change recall.
    if type(bright_only) is not bool:
        raise ValueError("Bright-only search must be explicit boolean")
    if rgb.shape != (240, 320, 3) or rgb.dtype != np.uint8:
        raise ValueError("Experimental search requires 320x240 RGB uint8")
    # Split once rather than copying interleaved slices for each OpenCV call.
    red, green, blue = cv2.split(rgb)
    planes = (cv2.subtract(red, cv2.max(green, blue)), cv2.subtract(blue, cv2.max(red, green)))
    pool = []
    for group, plane in enumerate(planes):
        detector = cv2.MSER_create(5, 6, 25000)
        if bright_only:
            detector.setPass2Only(True)
        regions, _ = detector.detectRegions(plane)
        for points in regions:
            reduced = native.reduce(plane, points)
            if len(reduced):
                c = candidate(reduced, group, rgb.shape)
                if c:
                    pool.append(c)
    return select(pool)
