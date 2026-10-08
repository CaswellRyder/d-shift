"""Experimental low-resolution search with original-detail crop verification.

Retains the same candidate geometry/budget. Cannot recover missing candidates.
No camera or actuation. Not enabled by default in any existing application.
"""
import time

import cv2
import numpy as np

from .vision import proposals, suppress_duplicates, validate_model_profile


def validate_rgb(rgb):
    if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3
            or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 8):
        raise ValueError("Expected HWC RGB uint8 image of at least 8x8")


def map_box(box, scale, shape):
    sx, sy = scale
    height, width = shape[:2]
    left, top, right, bottom = box
    return [max(0, int(np.floor(left*sx))), max(0, int(np.floor(top*sy))),
            min(width, int(np.ceil(right*sx))), min(height, int(np.ceil(bottom*sy)))]


def observe_detail(rgb, predictor, scan_rgb=None, scan_size=(320,240), limit=12,
                   profile="balloon_components", duplicate_policy="nested"):
    """Boxes use SOURCE pixels; normalized centers remain resolution independent.

    An optional scan_rgb must be the same frame/FOV as rgb (caller responsibility).
    This allows synchronized camera streams and exact frozen-proposal ablations.
    If omitted, search image is resized locally and its cost is included in timing.
    """
    validate_model_profile(predictor.metadata, profile)
    start = time.perf_counter()
    validate_rgb(rgb)
    if scan_rgb is None:
        if len(scan_size) != 2 or min(scan_size) < 8:
            raise ValueError("Invalid scan size")
        scan_rgb = cv2.resize(rgb, scan_size)
    validate_rgb(scan_rgb)
    height, width = rgb.shape[:2]
    sh, sw = scan_rgb.shape[:2]
    if sh > height or sw > width:
        raise ValueError("Detail source must not be smaller than search image")
    scale = (width/sw, height/sh)
    candidates = proposals(scan_rgb, predictor.metadata["task"], limit=limit, profile=profile)
    mapped = [dict(c, scan_box=c["box"], scan_crop_box=c["crop_box"],
                   box=map_box(c["box"], scale, rgb.shape),
                   crop_box=map_box(c["crop_box"], scale, rgb.shape),
                   area=c["area"]*scale[0]*scale[1]) for c in candidates]
    crops = [rgb[c["crop_box"][1]:c["crop_box"][3], c["crop_box"][0]:c["crop_box"][2]]
             for c in mapped]
    results = (predictor.predict_many(crops) if hasattr(predictor, "predict_many")
               else [predictor.predict(crop) for crop in crops])
    found = []
    for candidate, result in zip(mapped, results):
        left, top, right, bottom = candidate["box"]
        found.append(dict(candidate, **result,
                          center_normalized=[(left+right)/width-1, (top+bottom)/height-1]))
    observations = suppress_duplicates(found, duplicate_policy)
    return dict(observations=observations, accepted_count=sum(r["accepted"] for r in observations),
                suppressed_count=sum(r["suppressed"] for r in observations),
                coordinate_space="source_pixels", image_size=[width,height], scan_size=[sw,sh],
                proposal_profile=profile, duplicate_policy=duplicate_policy,
                processing_ms=(time.perf_counter()-start)*1000,
                flight_commands=None, deployment_approved=False)
