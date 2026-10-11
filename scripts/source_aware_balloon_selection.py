"""Research-only source-priority duplicate selection, never a confidence override."""
from dtr.tracking import iou
from scripts.research_confident_containment import select as incumbent_select
from scripts.research_balloon_confidence_selection import concentric
from scripts.compare_balloon_containment import nested


def conflict(a, b):
    return (a['label'] == b['label'] and
            (iou(a['box'], b['box']) > .5 or concentric(a['box'], b['box']) or nested(a['box'], b['box'])))


def select(found):
    """Keep each source's own winners; learned additions cannot evict MSER winners."""
    if len(found) > 12 or any(r.get('proposal_origin') not in ('mser', 'learned') for r in found):
        raise ValueError('Require explicit provenance and at most12 raw predictions')
    rows = [None] * len(found)
    for source in ('mser', 'learned'):
        indices = [i for i, row in enumerate(found) if row['proposal_origin'] == source]
        selected = incumbent_select([found[i] for i in indices])
        for i, row in zip(indices, selected, strict=True):
            for key in ('suppressed_by', 'suppressed_by_index'):
                if key in row:
                    row[key] = indices[row[key]]
            rows[i] = row
    accepted_base = [i for i, row in enumerate(rows) if row['proposal_origin'] == 'mser' and row['accepted']]
    for row in rows:
        if row['proposal_origin'] != 'learned' or not row['accepted']:
            continue
        winner = next((i for i in accepted_base if conflict(row, rows[i])), None)
        if winner is not None:
            row.update(accepted=False, suppressed=True, suppressed_by_index=winner,
                       suppression_reason='source_priority_duplicate')
    return rows
