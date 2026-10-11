"""Equal-dose background controls for12 new reviewed blue-object runtime crops."""
import numpy as np

from scripts.indoor5_trial_data import inputs as base_inputs
from scripts.blue_object_round2_data import checked_extra
from scripts.grid_clutter_data import load_crop


def choose(rows, extra, mode):
    if mode not in ('control', 'admit') or len(extra) != 12 or any(r['label'] != 'background' for r in extra):
        raise ValueError('Require declared arm and12 reviewed backgrounds')
    pool = [r for r in rows if r['label'] == 'background' and r['preserve']]
    if len(pool) < len(extra):
        raise ValueError('Insufficient matched background controls')
    picked = extra if mode == 'admit' else [pool[int(i)] for i in np.random.default_rng(47).choice(len(pool), len(extra), replace=False)]
    return [dict(r, preserve=False, view='original') for _ in range(2) for r in picked]


def inputs(mode):
    rows, labels, ordinary, tensors = base_inputs('admit')
    extra = checked_extra()
    if len(rows) != 1922 or int(ordinary.sum()) != 1602:
        raise ValueError('Shared indoor curriculum changed')
    if {r['sha256'] for r in rows} & {r['sha256'] for r in extra}:
        raise ValueError('New runtime crops repeat old training pixels')
    added = choose(rows, extra, mode)
    tensors[0] = np.concatenate((tensors[0], np.stack([load_crop(r)[0] for r in added])))
    tensors.append(np.stack([load_crop(r)[0] for r in extra]))
    return (rows + added, np.concatenate((labels, np.zeros(len(added), int))),
            np.concatenate((ordinary, np.zeros(len(added), bool))), tensors)
