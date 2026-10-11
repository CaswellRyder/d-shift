"""Equal-dose controls for eleven newly reviewed blue TRAIN crop views."""
import numpy as np

from scripts.blue_photo_expansion_data import inputs as base_inputs
from scripts.blue_balloon_batch_data import checked_extra
from scripts.grid_clutter_data import load_crop


def choose(rows, extra, mode):
    if mode not in ('control', 'admit') or len(extra) != 11:
        raise ValueError('Require declared arm and eleven reviewed positives')
    pool = [r for r in rows if r['label'] == 'blue_balloon' and r['preserve']]
    if len(pool) < len(extra):
        raise ValueError('Insufficient matched blue controls')
    picked = extra if mode == 'admit' else [pool[int(i)] for i in np.random.default_rng(47).choice(len(pool), len(extra), replace=False)]
    return [dict(r, preserve=False, view='original') for _ in range(2) for r in picked]


def inputs(mode):
    rows, labels, ordinary, tensors = base_inputs('admit')
    extra = checked_extra()
    if len(rows) != 1764 or int(ordinary.sum()) != 1602:
        raise ValueError('Frozen base curriculum changed')
    added = choose(rows, extra, mode)
    tensors[0] = np.concatenate((tensors[0], np.stack([load_crop(r)[0] for r in added])))
    tensors.append(np.stack([load_crop(r)[0] for r in extra]))
    return (rows + added, np.concatenate((labels, np.full(len(added), 2))),
            np.concatenate((ordinary, np.zeros(len(added), bool))), tensors)
