"""Colour/geometry proposals for all hoops, confirmed by the seven-class crop model.

The historical module and square search names remain importable for callers.
Geometry thresholds describe complete silhouettes, not calibrated flight limits.
"""

import time

import cv2
import numpy as np

from .vision import color_masks
from .goal_evidence import CLASSES, SHAPES, goal_evidence

GOAL_COLORS = ("orange", "yellow")  # Index matches the goal color_masks order.
SQUARE_LABELS = {"orange": "orange_square", "yellow": "yellow_square"}
GOAL_LABELS = tuple(f"{color}_{shape}" for color in GOAL_COLORS for shape in SHAPES)
LABELS_BY_COLOR = {color: tuple(f"{color}_{shape}" for shape in SHAPES) for color in GOAL_COLORS}
MIN_SIDE = 8
ASPECT_LIMITS = {"square": (1 / 1.8, 1.8), "circle": (0.6, 1.65),
                 "triangle": (0.6, 1.8)}  # Point-down triangle is nominally 1.15 w/h.
CORNER_EPS = 0.035  # Retains >=6 corners on circles, three on a triangle.
MIN_FILL = 0.65  # Hull area / min-area rectangle; triangles sit near 0.5.
# Thick frames and blur clip a far triangle's tips, so its hull often simplifies to 4-6
# corners. Test against the smallest enclosing triangle instead of a corner count:
# a triangle fills ~1.0 of it, a circle ~0.60 and a square ~0.50. Bounding-rectangle fill
# rises from 0.5 toward 0.7 as the frame thickens and blurs.
TRIANGLE_FILL = (0.40, 0.75)
MIN_TRIANGLE_RATIO = 0.80
MAX_TIP_GAP = 0.30  # Enclosing-vertex distance from the hull, fraction of mean side; far lab tips reach ~0.22.
MIN_CIRCULARITY = 0.82
MAX_PARALLEL_SINE = 0.25  # Opposite edge directions differ by at most ~14.5 degrees.
MIN_OPPOSITE_RATIO = 0.65
HOLLOW = 0.02  # A visible opening; hollow contours are classified first.
# Nominal outer width/height (m) from the DTR rules, used only to rank proposals by range
# before the crop budget; flight ranking reads configs/target_sizes.json.
NOMINAL_SIZES = {"circle": (1.130, 1.130), "square": (1.168, 1.168), "triangle": (1.749, 1.515)}
SMALL_SIDE = 16  # Rasterized circles and squares this small cannot be told apart reliably.
DEFAULT_MAX_CROPS = 4
DEFAULT_THRESHOLD = 0.8
DEFAULT_COLOR_THRESHOLD = 0.8


def order_corners(points):
    """Top-left, top-right, bottom-right, bottom-left in image coordinates."""
    points = np.asarray(points, np.float32).reshape(4, 2)
    total, diff = points.sum(1), points[:, 1] - points[:, 0]
    return [points[np.argmin(total)], points[np.argmin(diff)],
            points[np.argmax(total)], points[np.argmax(diff)]]


def shape_hint(hull):
    """Return shape and cyclic hull corners, rejecting fragments/trapezoids."""
    area, perimeter = cv2.contourArea(hull), cv2.arcLength(hull, True)
    _, (rw, rh), _ = cv2.minAreaRect(hull)
    if min(area, perimeter, rw, rh) <= 0:
        return None, None
    fill = area / (rw * rh)
    if TRIANGLE_FILL[0] <= fill <= TRIANGLE_FILL[1]:
        triangle_area, triangle = cv2.minEnclosingTriangle(hull.astype(np.float32))
        triangle = triangle.reshape(-1, 2)
        side = np.mean(np.linalg.norm(np.roll(triangle, -1, axis=0) - triangle, axis=1))
        # Clipped tips stay near the enclosing vertex; a trapezoid's missing apex does not.
        gaps = [-cv2.pointPolygonTest(hull, (float(x), float(y)), True) for x, y in triangle]
        if (triangle_area > 0 and area / triangle_area >= MIN_TRIANGLE_RATIO
                and max(gaps) <= MAX_TIP_GAP * side):
            return "triangle", triangle
    corners = cv2.approxPolyDP(hull, CORNER_EPS * perimeter, True).reshape(-1, 2)
    if len(corners) == 4 and fill >= MIN_FILL and max(rw / rh, rh / rw) <= 1.8:
        # Cyclic order avoids duplicate vertices from sum/difference corner ordering.
        edges = np.roll(corners.astype(float), -1, axis=0) - corners
        lengths = np.linalg.norm(edges, axis=1)
        for i in (0, 1):
            a, b = edges[i], edges[i + 2]
            sine = abs(a[0] * b[1] - a[1] * b[0]) / (lengths[i] * lengths[i + 2])
            ratio = min(lengths[i], lengths[i + 2]) / max(lengths[i], lengths[i + 2])
            if sine > MAX_PARALLEL_SINE or ratio < MIN_OPPOSITE_RATIO:
                return None, None
        return "square", corners
    if len(corners) >= 6 and fill >= MIN_FILL and 4 * np.pi * area / perimeter**2 >= MIN_CIRCULARITY:
        return "circle", corners
    return None, None


def goal_candidates(rgb, colors=GOAL_COLORS):
    """Outer contours in the requested colours: hollow, then nearest, alternating colours."""
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
            if min(w, h) < MIN_SIDE:
                continue
            if w * h > 0.75 * width * height:
                continue
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area <= 0:
                continue
            shape, corners = shape_hint(hull)
            if shape is None or not ASPECT_LIMITS[shape][0] <= w / h <= ASPECT_LIMITS[shape][1]:
                continue
            hole, child = 0.0, hierarchy[i][2]
            while child != -1:
                hole, child = max(hole, cv2.contourArea(contours[child])), hierarchy[child][0]
            pad = int(max(w, h) * 0.12)  # Same crop padding the goal model was trained on.
            # Small round/square blobs are ambiguous; let confident model evidence decide.
            shapes = (("circle", "square") if max(w, h) < SMALL_SIDE and shape != "triangle"
                      else (shape,))
            found.append(dict(
                box=[x, y, x + w, y + h],
                crop_box=[max(0, x - pad), max(0, y - pad),
                          min(width, x + w + pad), min(height, y + h + pad)],
                color=color,
                shape=shape,
                shapes=shapes,
                range_proxy=min(NOMINAL_SIZES[shape][0] / w, NOMINAL_SIZES[shape][1] / h),
                corners=corners.astype(float).tolist(),
                hollow=hole / hull_area,
                area=float(hull_area),
            ))
    # Hollow first, then nearest by physical size, alternating colours so the crop budget
    # covers both our goals and the opponent's.
    found.sort(key=lambda c: (c["hollow"] < HOLLOW, c["range_proxy"]))
    queues = [[c for c in found if c["color"] == color] for color in GOAL_COLORS]
    ordered = []
    while any(queues):
        for queue in queues:
            if queue:
                ordered.append(queue.pop(0))
    return ordered


def validate_goal_model(metadata):
    classes = (metadata or {}).get("classes", ())
    if (metadata or {}).get("task") != "goal" or len(classes) != 7 or set(classes) != set(CLASSES):
        raise ValueError("Goal search requires the seven-class goal model")


def find_goals(rgb, predictor, colors=GOAL_COLORS, threshold=DEFAULT_THRESHOLD,
                 max_crops=DEFAULT_MAX_CROPS, color_threshold=DEFAULT_COLOR_THRESHOLD):
    """Confirm matching shape and conditional colour evidence. Observational only."""
    validate_goal_model(predictor.metadata)
    if not np.isfinite(threshold) or not 0.5 < threshold <= 1:
        raise ValueError("Invalid shape threshold")
    if not isinstance(max_crops, int) or isinstance(max_crops, bool) or max_crops < 0:
        raise ValueError("max_crops must be a nonnegative integer")
    if not np.isfinite(color_threshold) or not 0.5 <= color_threshold <= 1:
        raise ValueError("Invalid colour threshold")
    classes = predictor.metadata["classes"]
    start = time.perf_counter()
    candidates = goal_candidates(rgb, colors)[:max_crops]
    searched = time.perf_counter()
    crops = [rgb[c["crop_box"][1]:c["crop_box"][3], c["crop_box"][0]:c["crop_box"][2]]
             for c in candidates]
    results = (predictor.predict_many(crops) if hasattr(predictor, "predict_many")
               else [predictor.predict(crop) for crop in crops])
    classified = time.perf_counter()
    height, width = rgb.shape[:2]
    detections = []
    for candidate, result in zip(candidates, results):
        evidence = goal_evidence(result["scores"], classes, threshold)
        shape, shape_score = evidence["shape"], evidence["shape_score"]
        scores = evidence["color_scores_given_shape"]
        color = max(scores, key=scores.get)
        color_score = scores[color]
        if (shape not in candidate["shapes"] or color_score < color_threshold
                or color not in colors or color != candidate["color"]):
            continue
        x1, y1, x2, y2 = candidate["box"]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        detections.append(dict(
            candidate, label=f"{color}_{shape}", color=color,
            score=float(shape_score), color_score=float(color_score),
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


# Legacy names now search all goal shapes. Flight callers use the explicit names.
square_candidates = goal_candidates
find_squares = find_goals
