"""Diagnostic direct detections from the tiny localizer; not the deployed runtime."""
import math

from scripts.research_blue_peak_blend import decode
from scripts.research_balloon_confidence_selection import confidence_selection
from scripts.compare_balloon_containment import area, nested


def select_direct(found, threshold):
    # Incumbent confidence_ordered assumes every accepted crop score is >=0.8.
    # Keep its geometry/order but validate against this direct detector's declared
    # threshold; never inflate scores to work around the crop-runtime contract.
    rows = confidence_selection(found)
    for row in rows:
        if row.get('suppression_reason') == 'confirmed_balloon_part':
            if row['accepted'] or not row['suppressed'] or row['score'] < threshold:
                raise ValueError('Invalid direct-detector part suppression')
            row.update(accepted=True, suppressed=False)
            row.pop('suppression_reason')
    accepted = sorted((i for i, row in enumerate(rows) if row['accepted']),
                      key=lambda i: (-rows[i]['score'], -area(rows[i]['box']), i))
    kept = []
    for i in accepted:
        winner = next((j for j in kept if rows[j]['label'] == rows[i]['label']
                       and nested(rows[j]['box'], rows[i]['box'])), None)
        if winner is None:
            kept.append(i)
        else:
            rows[i].update(accepted=False, suppressed=True, suppressed_by_index=winner,
                           suppression_reason='confidence_ordered_containment')
    return rows


def detect(values, threshold=.5):
    if not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError('Require a finite probability threshold')
    found = []
    for proposal in decode(values, True):
        group, probability = proposal['color_group'], proposal['proposal_score']
        scores = [1-probability, probability if group == 0 else 0., probability if group == 1 else 0.]
        found.append(dict(proposal, label=('red_balloon', 'blue_balloon')[group],
            score=probability, scores=scores, accepted=probability >= threshold))
    return select_direct(found, threshold)
