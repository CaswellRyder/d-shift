"""Fixed shared-IoU head composition, for offboard research inference only."""
import numpy as np

from scripts.audit_center_presence_localizer import classify, as_direct
from scripts.audit_direct_hybrid_balloon import raw_direct
from scripts.cached_balloon_decoder import decode
from scripts.direct_hybrid_balloon import merge
from scripts.shared_proposal_quality import feature_row, gate


def infer(rgb, base, grid, maps, classifier, head, mean, scale):
    if (np.asarray(mean).shape != (33,) or np.asarray(scale).shape != (33,)
            or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale < .001)):
        raise ValueError('Require frozen finite training normalization')
    crops = classify(classifier, rgb, base)
    direct = as_direct(decode(grid, False))
    parent = merge(crops, raw_direct(dict(detections=direct)), 'base_first')
    accepted = [row for row in parent if row['accepted']]
    values = []
    if accepted:
        fx = np.stack([feature_row(maps, row) for row in accepted])
        values = head((fx-mean)/scale, training=False).numpy()[:, 0].tolist()
    it = iter(values)
    selected = gate(parent, [next(it) if row['accepted'] else None for row in parent])
    return parent, selected, len(accepted)
