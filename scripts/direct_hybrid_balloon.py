"""Research merge of crop-classified MSER and direct localizer detections."""
from scripts.direct_localizer_balloon import select_direct
from scripts.research_confident_containment import select as select_crops
from scripts.source_aware_balloon_selection import conflict


def merge(base, direct, policy):
    if policy not in ('pooled', 'base_first') or len(base) > 12 or len(direct) > 12:
        raise ValueError('Require declared merge and bounded per-source candidates')
    for found, threshold in ((base, .8), (direct, .5)):
        if any(bool(r['accepted']) != (r['label'] != 'background' and r['score'] >= threshold) for r in found):
            raise ValueError('Source-specific acceptance threshold changed')
    a = [dict(row, proposal_origin='mser') for row in base]
    b = [dict(row, proposal_origin='learned') for row in direct]
    if policy == 'pooled':
        return select_direct(a+b, .5)
    a, b = select_crops(a), select_direct(b, .5)
    for row in b:
        for key in ('suppressed_by', 'suppressed_by_index'):
            if key in row:
                row[key] += len(a)
        if row['accepted']:
            winner = next((i for i, old in enumerate(a) if old['accepted'] and conflict(row, old)), None)
            if winner is not None:
                row.update(accepted=False, suppressed=True, suppressed_by_index=winner,
                           suppression_reason='source_priority_duplicate')
    return a+b
