"""Fixed-operating-point detector metrics. Development pass never approves flight."""

import math

from .tracking import iou


def size_bin(box, width, height):
    longest = max((box[2]-box[0])*320/width, (box[3]-box[1])*240/height)
    return "under16px" if longest < 16 else "16to31px" if longest < 32 else "32pluspx"


def match(truth, predictions, threshold, class_agnostic=False):
    used, hits = set(), []
    for pred in sorted(predictions, key=lambda p: -p["score"]):
        best = max((i for i, t in enumerate(truth) if i not in used
                    and (class_agnostic or t["label"] == pred["label"])),
                   key=lambda i: iou(pred["box"], truth[i]["box"]), default=None)
        if best is not None and iou(pred["box"], truth[best]["box"]) >= threshold:
            used.add(best)
        else:
            best = None
        hits.append((pred, best))
    return hits


def summarize(frames, standard):
    names = standard["classes"]
    counts = {k: dict(targets=0, true_positive=0, false_positive=0) for k in names}
    sizes = {k: {s: dict(targets=0, true_positive=0, false_positive=0)
                 for s in ("under16px", "16to31px", "32pluspx")} for k in names}
    localization = dict(targets=0, true_positive=0, false_positive=0)
    for frame in frames:
        truth = frame["truth"]
        width, height = frame["image_size"]
        if width <= 0 or height <= 0 or not all(math.isfinite(n) for n in (width, height)):
            raise ValueError("Invalid evaluation image dimensions")
        for row in [*truth, *frame["predictions"]]:
            box = row["box"]
            if row["label"] not in names or len(box) != 4 or not all(math.isfinite(n) for n in box):
                raise ValueError("Invalid evaluation label or box")
            if box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError("Non-positive evaluation box")
        for pred in frame["predictions"]:
            if not math.isfinite(pred["score"]) or not 0 <= pred["score"] <= 1:
                raise ValueError("Invalid detector score")
        preds = [p for p in frame["predictions"] if p["score"] >= standard["confidence_threshold"]]
        for t in truth:
            counts[t["label"]]["targets"] += 1
            sizes[t["label"]][size_bin(t["box"], width, height)]["targets"] += 1
        for pred, target in match(truth, preds, standard["iou_threshold"]):
            key = "true_positive" if target is not None else "false_positive"
            counts[pred["label"]][key] += 1
            box = truth[target]["box"] if target is not None else pred["box"]
            sizes[pred["label"]][size_bin(box, width, height)][key] += 1
        localization["targets"] += len(truth)
        for _, target in match(truth, preds, standard["iou_threshold"], class_agnostic=True):
            localization["true_positive" if target is not None else "false_positive"] += 1

    def metrics(c):
        tp, fp, n = c["true_positive"], c["false_positive"], c["targets"]
        return {**c, "precision": tp/max(1, tp+fp), "recall": tp/max(1, n),
                "f1": 2*tp/max(1, tp+fp+n)}

    classes = {k: metrics(c) for k, c in counts.items()}
    failed = []
    for k, c in classes.items():
        reasons = []
        if c["targets"] < standard["minimum_targets_per_class"]:
            reasons.append("insufficient_targets")
        if c["precision"] < standard["minimum_precision_per_class"]:
            reasons.append("precision")
        if c["recall"] < standard["minimum_recall_per_class"]:
            reasons.append("recall")
        if reasons:
            failed.append(dict(label=k, reasons=reasons))
    colors = {color: metrics({stat: sum(c[stat] for k, c in counts.items()
                                       if k.startswith(color + "_"))
                              for stat in ("targets", "true_positive", "false_positive")})
              for color in ("orange", "yellow")}
    return dict(classes=classes, colors=colors, class_agnostic_localization=metrics(localization),
                size_strata={k: {s: metrics(c) for s, c in rows.items()} for k, rows in sizes.items()},
                size_reference="Longest side in equivalent 320x240 frame coordinates",
                development_passed=bool(frames) and not failed, failed_requirements=failed,
                deployment_approved=False, test_evaluated=False)
