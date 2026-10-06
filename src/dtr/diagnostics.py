"""Annotation-relative pipeline triage. Diagnostics are not corrected labels."""

from .tracking import iou


def diagnose_frame(truth, observations, diagnostic_proposals=(), threshold=0.5):
    """Partition misses after confidence-ordered, same-class one-to-one matching.

    Miss causes follow the furthest successful stage, not the highest scoring crop.
    An accepted crop already assigned to another annotation is an assignment conflict.
    The larger proposal budget is geometry-only: it is never scored by the teacher.
    """
    used, matched, predictions = set(), {}, []
    for index in sorted(range(len(observations)), key=lambda i: -observations[i]["score"]):
        pred = observations[index]
        if not pred["accepted"]:
            continue
        target = max(
            (i for i, t in enumerate(truth) if i not in used and t["label"] == pred["label"]),
            key=lambda i: iou(pred["box"], truth[i]["box"]),
            default=None,
        )
        if target is not None and iou(pred["box"], truth[target]["box"]) >= threshold:
            used.add(target)
            matched[target] = index
            cause = "matched"
        else:
            overlaps = [t for t in truth if iou(pred["box"], t["box"]) >= threshold]
            if any(t["label"] == pred["label"] for t in overlaps):
                cause = "duplicate_or_assignment"
            elif overlaps:
                cause = "wrong_class"
            elif any(iou(pred["box"], t["box"]) > 0.1 for t in truth):
                cause = "partial_box"
            else:
                cause = "no_labeled_overlap"
            target = None
        predictions.append(dict(index=index, cause=cause, target=target))

    targets = []
    for index, actual in enumerate(truth):
        covered = [
            i for i, p in enumerate(observations) if iou(p["box"], actual["box"]) >= threshold
        ]
        correct = [i for i in covered if observations[i]["label"] == actual["label"]]
        if index in matched:
            cause = "detected"
        elif any(observations[i]["accepted"] for i in correct):
            cause = "assignment_conflict"
        elif any(observations[i].get("raw_accepted", False) for i in correct):
            cause = "suppressed"
        elif correct:
            cause = "below_threshold"
        elif covered:
            cause = (
                "wrong_class"
                if any(observations[i]["label"] != "background" for i in covered)
                else "background"
            )
        elif any(iou(p["box"], actual["box"]) >= threshold for p in diagnostic_proposals):
            cause = "proposal_budget"
        else:
            cause = "localization"
        targets.append(
            dict(
                index=index,
                label=actual["label"],
                cause=cause,
                covered_by=covered,
                best_iou=max((iou(p["box"], actual["box"]) for p in observations), default=0),
                matched_by=matched.get(index),
            )
        )
    return dict(targets=targets, predictions=predictions)
