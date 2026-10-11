"""Fixed-budget TRAIN experiment; do not change deployed candidate selection."""
from dtr.tracking import iou
from scripts.audit_balloon_dense_proposals import supplement


def reserved(base, learned):
    """Use up to two unique blue learned proposals, even when pixel slots are full."""
    if len(base) > 12:
        raise ValueError('Pixel proposal budget exceeded')
    if len(base) <= 10:
        return supplement(base, learned)
    added = []
    for row in learned:
        if row['color_group'] == 1 and not any(iou(row['box'], r['box']) > .65 for r in base+added):
            added.append(row)
        if len(added) == 2:
            break
    return base[:12-len(added)]+added
