"""Separate joint goal scores into shape/color evidence, without relabeling.

Scores are model evidence, not calibrated probabilities or independent sensor
measurements. This does not find candidates rejected by the HSV proposal stage.
"""
import math

SHAPES = ("circle", "square", "triangle")
COLORS = ("orange", "yellow")
CLASSES = ("background",) + tuple(f"{c}_{s}" for c in COLORS for s in SHAPES)


def goal_evidence(scores, classes, threshold=.8):
    if len(classes) != 7 or set(classes) != set(CLASSES):
        raise ValueError("Require the seven-class goal taxonomy")
    values = [float(s) for s in scores]
    if (len(values) != len(classes) or any(not math.isfinite(s) or not 0 <= s <= 1 for s in values)
            or not math.isclose(sum(values),1,abs_tol=1e-5)
            or not math.isfinite(threshold) or not .5 < threshold <= 1):
        raise ValueError("Invalid goal scores or threshold")
    by_class = dict(zip(classes,values))
    shape_scores = {s:sum(by_class[f"{c}_{s}"] for c in COLORS) for s in SHAPES}
    best_shape = max(shape_scores,key=shape_scores.get)
    shape_score = shape_scores[best_shape]
    shape_known = shape_score >= threshold
    # Conditional on the SAME shape, not on all goals: avoid combining yellow
    # square evidence with unrelated orange triangle evidence.
    color_scores = {c:by_class[f"{c}_{best_shape}"]/shape_score if shape_score else 0 for c in COLORS}
    best_color = max(color_scores,key=color_scores.get)
    color = best_color if shape_known and color_scores[best_color] >= threshold else "unknown"
    return dict(shape=best_shape if shape_known else "unknown", shape_score=shape_score,
                shape_scores=shape_scores, goal_score=1-by_class["background"],
                color=color, color_scores_given_shape=color_scores,
                status="shape_and_color" if color != "unknown" else "shape_only" if shape_known else "uncertain",
                threshold=threshold, calibrated=False, evidence_source="joint_class_score_marginals")


def goal_candidate(observation, target_color, *, measurement_age_ms,
                   allow_unknown_color=False, max_age_ms=500):
    """Opt-in investigation policy, NOT movement/entry/capture authorization.

Measurement age must come from capture timing, not cached classification age.
Unknown color can be investigated symmetrically for either selected goal color.
It never becomes a confirmed orange/yellow label by exclusion.
"""
    if target_color not in COLORS:
        raise ValueError("Target color must be orange or yellow")
    if not math.isfinite(max_age_ms) or max_age_ms <= 0:
        raise ValueError("Invalid maximum measurement age")
    result = dict(target_color=target_color, confirmed_target=False,
                  investigation_candidate=False, action_authorized=False, flight_commands=None)
    age = measurement_age_ms
    class_age = observation.get("classification_age_ms",age)
    if (age is None or class_age is None or not math.isfinite(age) or not math.isfinite(class_age)
            or not 0 <= age <= max_age_ms or not 0 <= class_age <= 1000
            or observation.get("suppressed",False) or observation.get("tracking_valid") is False):
        return dict(result,status="invalid_or_stale")
    evidence = observation.get("goal_evidence")
    if not evidence or evidence["shape"] == "unknown":
        return dict(result,status="insufficient_shape_evidence")
    color = evidence["color"]
    if color not in (target_color,"unknown"):
        return dict(result,status="other_color")
    if (color == target_color and observation.get("accepted",False)
            and observation.get("label") == f'{target_color}_{evidence["shape"]}'):
        return dict(result,status="confirmed_target",confirmed_target=True)
    if color == "unknown" and allow_unknown_color:
        return dict(result,status="tentative_color_unknown",investigation_candidate=True)
    return dict(result,status="shape_only_unconfirmed")
