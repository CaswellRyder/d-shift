"""Bounded OpenCV color proposals + selected crop verifier; observational only."""

import time

import cv2
import numpy as np

from .runtime import Predictor

HSV_RANGES = {
    "balloon": [((32, 30, 30), (95, 255, 255)), ((95, 35, 25), (175, 255, 255))],
    "goal": [((0, 45, 30), (12, 255, 255)), ((24, 50, 50), (42, 255, 255))],
}
DEFAULT_LIMIT = 12  # Offboard research budget, NOT an ARMv6 performance claim.
DEFAULT_PROFILE = "v2"
DEFAULT_DUPLICATE_POLICY = "nested"
RED_BLUE_PROFILE = "balloon_red_blue"
RED_BLUE_CLASSES = frozenset(("background", "red_balloon", "blue_balloon"))
PROPOSAL_PROFILES = (
    "v2",
    "orange_v3",
    "orange_low_sat",
    "rim_v4",
    "rim_budget",
    "orange_local",
    "balloon_external",
    "balloon_components",
    RED_BLUE_PROFILE,
    "goal_gap9",
    "orange_regions",
    "orange_regions_compact",
    "goal_edges",
    "goal_lut",
)


def validate_model_profile(metadata, profile):
    """Never reinterpret historical green/purple outputs as red/blue identities."""
    metadata = metadata or {}
    classes = metadata.get("classes", ())
    red_blue_model = bool({"red_balloon", "blue_balloon"}.intersection(classes))
    if profile == RED_BLUE_PROFILE:
        if (metadata.get("task") != "balloon" or len(classes) != 3
                or set(classes) != RED_BLUE_CLASSES):
            raise ValueError("balloon_red_blue requires a red/blue balloon model, not legacy weights")
    elif red_blue_model:
        raise ValueError("Red/blue balloon models require the balloon_red_blue proposal profile")


def color_masks(rgb, task, profile=DEFAULT_PROFILE):
    if profile not in PROPOSAL_PROFILES:
        raise ValueError("Unknown proposal profile")
    if task not in HSV_RANGES:
        raise ValueError("Task must be balloon or goal")
    if profile == RED_BLUE_PROFILE and task != "balloon":
        raise ValueError("balloon_red_blue is only valid for the balloon task")
    if task == "goal" and profile == "goal_lut":
        from .color_lookup import goal_lookup_masks
        return goal_lookup_masks(rgb)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    if profile == RED_BLUE_PROFILE:
        # Broad proposal bands, not color decisions. The crop model must reject
        # orange goals, purple clutter, and non-balloon red/blue objects.
        red = cv2.inRange(hsv, (0, 45, 30), (10, 255, 255))
        red |= cv2.inRange(hsv, (170, 45, 30), (179, 255, 255))
        blue = cv2.inRange(hsv, (95, 45, 30), (130, 255, 255))
        return [red, blue]
    masks = [
        cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        for lo, hi in HSV_RANGES[task]
    ]
    if task == "goal":
        masks[0] |= cv2.inRange(
            hsv, np.array((170, 45, 30), np.uint8), np.array((179, 255, 255), np.uint8)
        )
        if profile in ("orange_v3", "orange_low_sat"):
            # Distant, blurred orange rims lose saturation and shift toward yellow.
            # Keep yellow separate; the teacher still decides shape/color identity.
            masks[0] |= cv2.inRange(
                hsv,
                np.array((0, 25, 40), np.uint8),
                np.array((23 if profile == "orange_v3" else 12, 255, 255), np.uint8),
            )
    return masks


def color_mask(rgb, task, profile=DEFAULT_PROFILE):
    # Do not erode away small balloons and thin goal rims before proposing boxes.
    combined = np.bitwise_or.reduce(color_masks(rgb, task, profile))
    if task == "goal" and profile == "orange_local":
        combined |= orange_local_mask(rgb)
    return combined


def orange_local_mask(rgb):
    """Experimental local red contrast; separate contours avoid merging baseline rims."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    possible = cv2.inRange(hsv, np.array((0, 25, 40), np.uint8), np.array((23, 255, 255), np.uint8))
    red_excess = cv2.subtract(rgb[:, :, 0], rgb[:, :, 1])
    local = cv2.morphologyEx(red_excess, cv2.MORPH_TOPHAT, np.ones((9, 9), np.uint8))
    return possible & np.where(local >= 4, 255, 0).astype(np.uint8)


def orange_regions(rgb):
    """Train-first small-region proposals; no learned identity or flight geometry.

    MSER searches multiple thresholds of red chroma instead of one global HSV cutoff.
    Local contrast rejects flat warm surfaces; size bounds target distant goal rims.
    """
    height, width = rgb.shape[:2]
    if min(height, width) < 3:
        return []
    values = rgb.astype(np.int16)
    chroma = np.clip(values[:, :, 0] - (values[:, :, 1] + values[:, :, 2]) // 2, 0, 255)
    chroma = chroma.astype(np.uint8)
    regions, boxes = cv2.MSER_create(3, 6, 1024).detectRegions(chroma)
    found = []
    for points, (x, y, w, h) in zip(regions, boxes):
        x, y, w, h = map(int, (x, y, w, h))
        if min(w, h) < 4 or max(w, h) > 32 or max(w / h, h / w) > 3:
            continue
        pixels = values[points[:, 1], points[:, 0]]
        strength = float(chroma[points[:, 1], points[:, 0]].mean())
        if strength < 8 or float((pixels[:, 0] - pixels[:, 1]).mean()) < 3:
            continue
        left, top, right, bottom = max(0, x-2), max(0, y-2), min(width, x+w+2), min(height, y+h+2)
        window = chroma[top:bottom, left:right]
        ring = np.ones(window.shape, dtype=bool)
        ring[y-top:y-top+h, x-left:x-left+w] = False
        if not ring.any() or strength - float(window[ring].mean()) < 3:
            continue
        # The convex hull estimates a broken rim's extent, not its aperture.
        area = float(cv2.contourArea(cv2.convexHull(points)))
        if area < 6:
            continue
        pad = int(max(w, h) * 0.12)
        found.append(dict(
            box=[x, y, x+w, y+h],
            crop_box=[max(0, x-pad), max(0, y-pad), min(width, x+w+pad), min(height, y+h+pad)],
            area=area, color_group=0,
            proposal_score=float(np.sqrt(area) * (area / (w*h)) ** 2),
            proposal_source="orange_regions",
        ))
    return found


def nested_rim(outer, inner):
    """Near-concentric boxes of similar scale; not arbitrary object containment."""
    a, b = outer, inner
    large = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    small = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    return (
        large >= small > 0
        and small >= 0.35 * large
        and intersection / small >= 0.9
        and abs(a[0] + a[2] - b[0] - b[2]) <= 0.3 * (a[2] - a[0])
        and abs(a[1] + a[3] - b[1] - b[3]) <= 0.3 * (a[3] - a[1])
    )


def consolidate_rims(candidates, conservative=False):
    """Free proposal slots before inference; only merge within a color group."""
    kept = []
    for candidate in sorted(
        candidates, key=lambda c: -((c["box"][2] - c["box"][0]) * (c["box"][3] - c["box"][1]))
    ):
        b = candidate["box"]
        matched = next(
            (
                c
                for c in kept
                if c["color_group"] == candidate["color_group"]
                and nested_rim(c["box"], b)
                and (
                    not conservative
                    or (b[2] - b[0]) * (b[3] - b[1])
                    >= 0.6 * (c["box"][2] - c["box"][0]) * (c["box"][3] - c["box"][1])
                )
            ),
            None,
        )
        if matched is not None:
            if conservative:
                matched["proposal_score"] = max(
                    matched["proposal_score"], candidate["proposal_score"]
                )
            continue
        kept.append(dict(candidate))
    return kept


def proposals(rgb, task, limit=DEFAULT_LIMIT, min_area=6, profile=DEFAULT_PROFILE):
    from .tracking import iou

    if not isinstance(limit, int) or not 1 <= limit <= 64:
        raise ValueError("Proposal limit must be between 1 and 64")
    if profile == "goal_lut" and task == "balloon":
        profile = "balloon_components"
    if profile == "goal_edges":
        baseline = proposals(rgb, task, limit, min_area, profile="balloon_components")
        if task != "goal":
            return baseline
        from .goal_search import mixed_candidates
        return mixed_candidates(rgb, baseline, limit)
    candidates = []
    height, width = rgb.shape[:2]
    masks = list(enumerate(color_masks(rgb, task, profile)))
    if task == "goal" and profile == "orange_local":
        masks.append((0, orange_local_mask(rgb)))
    for color, raw in masks:
        kernels = (1, 3) if task == "balloon" else (1, 5)
        if task == "goal" and profile == "goal_gap9":
            # Train-first ablation: bridge wider breaks without expanding HSV bands.
            kernels = (1, 5, 9)
        for kernel in kernels:
            mask = (
                raw
                if kernel == 1
                else cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((kernel, kernel), np.uint8))
            )
            retrieval = (
                cv2.RETR_EXTERNAL
                if task == "balloon" and profile == "balloon_external"
                else cv2.RETR_LIST
            )
            if task == "balloon" and profile in (
                "balloon_components", "goal_gap9", "orange_regions", "orange_regions_compact",
                RED_BLUE_PROFILE,
            ):
                retrieval = cv2.RETR_CCOMP
            contours, hierarchy = cv2.findContours(mask, retrieval, cv2.CHAIN_APPROX_SIMPLE)
            for contour_index, contour in enumerate(contours):
                if min_area > 0 and len(contour) < 3:
                    continue  # Fewer than three vertices cannot enclose positive area.
                # CCOMP keeps foreground islands inside hoops, unlike RETR_EXTERNAL.
                # Child contours are holes in a component, not additional balloons.
                if retrieval == cv2.RETR_CCOMP and hierarchy[0][contour_index][3] != -1:
                    continue
                area = cv2.contourArea(contour)
                # Reject tiny contours before paying for a second native call.
                # This preserves the existing selection rule exactly.
                if area < min_area:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                if min(w, h) < 4 or max(w / h, h / w) > 5:
                    continue
                if w * h > 0.75 * width * height:
                    continue
                pad = int(max(w, h) * 0.12)
                candidates.append(
                    {
                        "box": [x, y, x + w, y + h],
                        "crop_box": [
                            max(0, x - pad),
                            max(0, y - pad),
                            min(width, x + w + pad),
                            min(height, y + h + pad),
                        ],
                        "area": float(area),
                        "color_group": color,
                        "proposal_score": float(np.sqrt(area) * (area / (w * h)) ** 2),
                    }
                )
    if task == "goal" and profile in ("orange_regions", "orange_regions_compact"):
        candidates.extend(orange_regions(rgb))
    if task == "goal" and profile in ("rim_v4", "rim_budget"):
        candidates = consolidate_rims(candidates, conservative=profile == "rim_budget")
    selected = []
    queues = [
        iter(
            sorted(
                (c for c in candidates if c["color_group"] == color),
                key=lambda c: -c["proposal_score"],
            )
        )
        for color in (0, 1)
    ]
    # Prevent large yellow structures from consuming the entire orange-goal budget.
    while len(selected) < limit:
        added = False
        for queue in queues:
            for candidate in queue:
                if any(
                    iou(candidate["box"], previous["box"]) > (
                        0.5 if task == "goal" and profile == "orange_regions_compact"
                        and candidate["color_group"] == previous["color_group"] == 0 else 0.65
                    ) for previous in selected
                ):
                    continue
                selected.append(candidate)
                added = True
                break
            if len(selected) == limit:
                break
        if not added:
            break
    return selected


def suppress_duplicates(observations, policy="nested"):
    """Keep auditable candidates, reject same-class nested rims and overlapping duplicates.

    Preserve the outer accepted box of concentric rims, even if the inner crop has
    a higher class score. Otherwise use confidence-ordered IoU suppression. This is
    a geometric heuristic, not instance segmentation; occluded targets can merge.
    """
    from .tracking import iou

    if policy not in ("none", "nested"):
        raise ValueError("Unknown duplicate policy")
    rows = [{**r, "raw_accepted": bool(r["accepted"]), "suppressed": False} for r in observations]
    if policy == "none":
        return rows

    def area(row):
        x1, y1, x2, y2 = row["box"]
        return max(0, x2 - x1) * max(0, y2 - y1)

    def reject(index, winner):
        rows[index].update(
            accepted=False, suppressed=True, suppressed_by=winner, rejection_reason="duplicate"
        )

    # Area-first only for near-concentric containment: inner and outer rim contours.
    order = sorted((i for i, r in enumerate(rows) if r["accepted"]), key=lambda i: -area(rows[i]))
    for position, outer in enumerate(order):
        if not rows[outer]["accepted"]:
            continue
        a = rows[outer]["box"]
        for inner in order[position + 1 :]:
            if not rows[inner]["accepted"] or rows[inner]["label"] != rows[outer]["label"]:
                continue
            b = rows[inner]["box"]
            intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
                0, min(a[3], b[3]) - max(a[1], b[1])
            )
            near_center = abs(a[0] + a[2] - b[0] - b[2]) <= 0.3 * (a[2] - a[0]) and abs(
                a[1] + a[3] - b[1] - b[3]
            ) <= 0.3 * (a[3] - a[1])
            if (
                area(rows[inner]) >= 0.35 * area(rows[outer])
                and area(rows[inner]) > 0
                and intersection / area(rows[inner]) >= 0.9
                and near_center
            ):
                reject(inner, outer)
    kept = []
    for index in sorted(
        (i for i, r in enumerate(rows) if r["accepted"]), key=lambda i: -rows[i]["score"]
    ):
        winner = next(
            (
                j
                for j in kept
                if rows[j]["label"] == rows[index]["label"]
                and iou(rows[j]["box"], rows[index]["box"]) > 0.5
            ),
            None,
        )
        if winner is None:
            kept.append(index)
        else:
            reject(index, winner)
    for row in rows:
        if row["suppressed"]:
            winner = row["suppressed_by"]
            while rows[winner]["suppressed"]:
                winner = rows[winner]["suppressed_by"]
            row["suppressed_by"] = winner
    return rows


def observe(
    rgb,
    predictor,
    limit=DEFAULT_LIMIT,
    profile=DEFAULT_PROFILE,
    duplicate_policy=DEFAULT_DUPLICATE_POLICY,
):
    """Scores are not calibrated probabilities. Boxes are not goal opening geometry."""
    validate_model_profile(predictor.metadata, profile)
    start = time.perf_counter()
    found = []
    candidates = proposals(rgb, predictor.metadata["task"], limit, profile=profile)
    searched_at = time.perf_counter()
    crops = [
        rgb[c["crop_box"][1] : c["crop_box"][3], c["crop_box"][0] : c["crop_box"][2]]
        for c in candidates
    ]
    results = (
        predictor.predict_many(crops)
        if hasattr(predictor, "predict_many")
        else [predictor.predict(crop) for crop in crops]
    )
    classified_at = time.perf_counter()
    for candidate, result in zip(candidates, results):
        x1, y1, x2, y2 = candidate["box"]
        found.append(
            {
                **candidate,
                **result,
                "center_normalized": [(x1 + x2) / rgb.shape[1] - 1, (y1 + y2) / rgb.shape[0] - 1],
            }
        )
    observations = suppress_duplicates(found, duplicate_policy)
    completed_at = time.perf_counter()
    return {
        "observations": observations,
        "accepted_count": sum(r["accepted"] for r in observations),
        "suppressed_count": sum(r["suppressed"] for r in observations),
        "proposal_profile": profile,
        "duplicate_policy": duplicate_policy,
        "processing_ms": (completed_at - start) * 1000,
        "stage_ms": {
            "search": (searched_at - start) * 1000,
            "crop_and_classify": (classified_at - searched_at) * 1000,
            "assemble_and_suppress": (completed_at - classified_at) * 1000,
        },
        "flight_commands": None,
        "deployment_approved": False,
    }


def replay(video, model, output, limit=3, max_frames=100, allow_unvalidated=False):
    import json
    from pathlib import Path

    if max_frames < 1 or limit < 1:
        raise ValueError("Frame and proposal limits must be positive")
    predictor = Predictor(model, allow_unvalidated=allow_unvalidated)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video}")
    dest = Path(output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    try:
        with dest.open("x") as stream:
            while count < max_frames:
                ok, bgr = cap.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(cv2.resize(bgr, (320, 240)), cv2.COLOR_BGR2RGB)
                result = observe(rgb, predictor, limit)
                result.update(frame=count, video_position_ms=cap.get(cv2.CAP_PROP_POS_MSEC))
                stream.write(json.dumps(result) + "\n")
                count += 1
    finally:
        cap.release()
    return {"frames": count, "output": str(dest), "mode": "offline replay; no actuation"}
