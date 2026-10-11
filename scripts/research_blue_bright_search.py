"""Research asymmetric search: full red, second MSER pass for blue only.

This changes proposals and needs accuracy evaluation. Geometry prefilter uses
OpenCV's own region bounding boxes and repeats existing rejection thresholds.
"""
import cv2
import numpy as np

from scripts.research_balloon_search import candidate, select


def impossible_geometry(width, height):
    return width < 4 or height < 4 or width > 3*height or height > 3*width or width*height > 57600


def search(rgb, native, blue_bright):
    if rgb.shape != (240, 320, 3) or rgb.dtype != np.uint8 or type(blue_bright) is not bool:
        raise ValueError('Require320x240 RGB uint8 and explicit policy boolean')
    red, green, blue = cv2.split(rgb)
    pool = []
    for group, plane in enumerate((cv2.subtract(red, cv2.max(green, blue)),
                                   cv2.subtract(blue, cv2.max(red, green)))):
        detector = cv2.MSER_create(5, 6, 25000)
        if group == 1 and blue_bright:
            detector.setPass2Only(True)
        regions, bounds = detector.detectRegions(plane)
        for points, (_, _, width, height) in zip(regions, bounds, strict=True):
            if impossible_geometry(int(width), int(height)):
                continue
            reduced = native.reduce(plane, points)
            if len(reduced):
                proposal = candidate(reduced, group, rgb.shape)
                if proposal:
                    pool.append(proposal)
    return select(pool)
