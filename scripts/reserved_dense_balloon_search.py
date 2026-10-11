"""Fixed two-slot learned proposal reservation; still at most12 classifications."""
from dtr.tracking import iou


def reserve(base, learned):
    """Keep first10 base candidates, admit up to2 novel learned boxes, refill unused slots."""
    if len(base) > 12 or len(learned) > 12:
        raise ValueError('Require existing12candidate decoder/search contracts')
    selected = list(base[:10])
    added = 0
    for row in learned:
        if added == 2 or len(selected) == 12:
            break
        if not any(iou(row['box'], p['box']) > .65 for p in selected):
            selected.append(row)
            added += 1
    for row in base[10:]:
        if len(selected) == 12:
            break
        if not any(iou(row['box'], p['box']) > .65 for p in selected):
            selected.append(row)
    return selected
