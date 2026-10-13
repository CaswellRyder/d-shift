"""Square-goal search: color + four-corner shape filter, then the goal crop model confirms.

Team strategy aims only at the square hoop (largest opening). The shape filter
replaces the 12-crop goal search, so the model sees well under one crop per frame
on DTR frames. Our goal color changes every round, so callers pass it per launch.
Settings were chosen on DTR v10 train (2026-10-13); val was checked once.
"""

import time

import cv2
import numpy as np

from .vision import color_masks

GOAL_COLORS = ("orange", "yellow")  # Index matches the goal color_masks order.
SQUARE_LABELS = {"orange": "orange_square", "yellow": "yellow_square"}
MIN_SIDE = 8  # Below ~16 px squares are mostly lost at the mask anyway.
MAX_ASPECT = 1.8  # Perspective squashes a hanging square; keep moderate skew.
CORNER_EPS = 0.06  # approxPolyDP tolerance, fraction of hull perimeter.
MIN_FILL = 0.65  # Hull area / min-area rectangle; triangles sit near 0.5.
HOLLOW = 0.02  # A visible opening; hollow quads are classified first.
DEFAULT_MAX_CROPS = 4
DEFAULT_THRESHOLD = 0.8


def order_corners(points):
    """Top-left, top-right, bottom-right, bottom-left in image coordinates."""
    points = np.asarray(points, np.float32).reshape(4, 2)
    total, diff = points.sum(1), points[:, 1] - points[:, 0]
    return [points[np.argmin(total)], points[np.argmin(diff)],
            points[np.argmax(total)], points[np.argmax(diff)]]


def square_candidates(rgb, colors=GOAL_COLORS):
    """Four-corner outer contours in the requested goal colors, hollow and large first."""
    unknown = set(colors) - set(GOAL_COLORS)
    if unknown or not colors:
        raise ValueError(f"Goal colors must be drawn from {GOAL_COLORS}")
    height, width = rgb.shape[:2]
    found = []
    for index, raw in enumerate(color_masks(rgb, "goal", "v2")):
        color = GOAL_COLORS[index]
        if color not in colors:
            continue
        mask = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            continue
        hierarchy = hierarchy[0]
        for i, contour in enumerate(contours):
            if hierarchy[i][3] != -1:
                continue  # Holes are read from the parent below.
            x, y, w, h = cv2.boundingRect(contour)
            if min(w, h) < MIN_SIDE or max(w / h, h / w) > MAX_ASPECT:
                continue
            if w * h > 0.75 * width * height:
                continue
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area <= 0:
                continue
            corners = cv2.approxPolyDP(hull, CORNER_EPS * cv2.arcLength(hull, True), True)
            if len(corners) != 4:
                continue
            _, (rw, rh), _ = cv2.minAreaRect(hull)
            if rw * rh <= 0 or hull_area / (rw * rh) < MIN_FILL:
                continue
            hole, child = 0.0, hierarchy[i][2]
            while child != -1:
                hole, child = max(hole, cv2.contourArea(contours[child])), hierarchy[child][0]
            pad = int(max(w, h) * 0.12)  # Same crop padding the goal model was trained on.
            found.append(dict(
                box=[x, y, x + w, y + h],
                crop_box=[max(0, x - pad), max(0, y - pad),
                          min(width, x + w + pad), min(height, y + h + pad)],
                color=color,
                corners=[[float(px), float(py)] for px, py in order_corners(corners)],
                hollow=hole / hull_area,
                area=float(hull_area),
            ))
    found.sort(key=lambda c: (c["hollow"] < HOLLOW, -c["area"]))
    return found


def validate_goal_model(metadata):
    classes = (metadata or {}).get("classes", ())
    if (metadata or {}).get("task") != "goal" or not set(SQUARE_LABELS.values()) <= set(classes):
        raise ValueError("Square-goal search requires the goal model with square classes")


def find_squares(rgb, predictor, colors=GOAL_COLORS, threshold=DEFAULT_THRESHOLD,
                 max_crops=DEFAULT_MAX_CROPS):
    """Confirmed squares of the requested colors, largest first. Observational only."""
    validate_goal_model(predictor.metadata)
    classes = predictor.metadata["classes"]
    start = time.perf_counter()
    candidates = square_candidates(rgb, colors)[:max_crops]
    searched = time.perf_counter()
    crops = [rgb[c["crop_box"][1]:c["crop_box"][3], c["crop_box"][0]:c["crop_box"][2]]
             for c in candidates]
    results = (predictor.predict_many(crops) if hasattr(predictor, "predict_many")
               else [predictor.predict(crop) for crop in crops])
    classified = time.perf_counter()
    height, width = rgb.shape[:2]
    detections = []
    for candidate, result in zip(candidates, results):
        scores = {color: result["scores"][classes.index(label)]
                  for color, label in SQUARE_LABELS.items()}
        # Shape evidence is the summed square mass; the model picks the color.
        color = max(scores, key=scores.get)
        if sum(scores.values()) < threshold or color not in colors:
            continue
        x1, y1, x2, y2 = candidate["box"]
        cx, cy = np.mean(candidate["corners"], axis=0)
        detections.append(dict(
            candidate, label=SQUARE_LABELS[color], color=color,
            score=float(sum(scores.values())),
            center_normalized=[float(2 * cx / width - 1), float(2 * cy / height - 1)],
            size_px=float(max(x2 - x1, y2 - y1)),
        ))
    return dict(
        detections=detections,
        neural_calls=len(crops),
        search_ms=(searched - start) * 1000,
        inference_ms=(classified - searched) * 1000,
        processing_ms=(time.perf_counter() - start) * 1000,
    )
