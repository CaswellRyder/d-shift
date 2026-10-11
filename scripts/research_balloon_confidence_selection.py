"""Research balloon selector: replace goal-rim area preference by class confidence.

Uses existing fixed containment/IoU thresholds, not fitted geometry. Confidence
is not localization quality; evaluate all regressions before any promotion.
"""
import math

from dtr.tracking import iou
from scripts.research_balloon_search import split_merged_balloon_parents, suppress_confirmed_balloon_parts


def area(box):
    return (box[2]-box[0])*(box[3]-box[1])


def concentric(a, b):
    outer, inner = (a, b) if area(a) >= area(b) else (b, a)
    overlap = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    return (area(inner) >= .35*area(outer) and overlap >= .9*area(inner)
        and abs(a[0]+a[2]-b[0]-b[2]) <= .3*(outer[2]-outer[0])
        and abs(a[1]+a[3]-b[1]-b[3]) <= .3*(outer[3]-outer[1]))


def confidence_selection(found):
    """Keep scores/candidates; change only the winner of competing accepted boxes."""
    rows = []
    for row in found:
        box = row['box']
        if (len(box) != 4 or not all(math.isfinite(v) for v in box)
                or box[2] <= box[0] or box[3] <= box[1]
                or not math.isfinite(row['score']) or not 0 <= row['score'] <= 1
                or row['label'] not in ('background', 'red_balloon', 'blue_balloon')):
            raise ValueError('Require finite balloon/background predictions and valid geometry')
        if any(k in row for k in ('raw_accepted', 'suppressed', 'suppressed_by', 'rejection_reason', 'suppression_reason')):
            raise ValueError('Require raw predictions, not previously selected observations')
        rows.append(dict(row, raw_accepted=bool(row['accepted']), suppressed=False))
    # Stable ties retain proposal order, as in existing confidence-ordered NMS.
    kept = []
    for i in sorted((i for i, r in enumerate(rows) if r['accepted']), key=lambda i: -rows[i]['score']):
        winner = next((j for j in kept if rows[j]['label'] == rows[i]['label']
            and (concentric(rows[j]['box'], rows[i]['box']) or iou(rows[j]['box'], rows[i]['box']) > .5)), None)
        if winner is None:
            kept.append(i)
        else:
            rows[i].update(accepted=False, suppressed=True, suppressed_by=winner,
                rejection_reason='confidence_balloon_duplicate')
    rows = split_merged_balloon_parents(rows, max_aspect=2.42913421076193)
    return suppress_confirmed_balloon_parts(rows)
