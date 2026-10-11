"""Equal class-count indoor additions versus resampling existing TRAIN crops."""
from collections import Counter

import numpy as np

from scripts.blue_batch_trial_data import inputs as base_inputs
from scripts.indoor_expansion5_data import checked_extra
from scripts.grid_clutter_data import load_crop

CLASSES = ('background', 'red_balloon', 'blue_balloon')


def choose(rows, extra, mode):
    if mode not in ('control', 'admit') or Counter(r['label'] for r in extra) != {
            'background': 11, 'red_balloon': 32, 'blue_balloon': 25}:
        raise ValueError('Require declared arm and reviewed68-row class counts')
    picked = list(extra)
    if mode == 'control':
        rng = np.random.default_rng(47)
        for label in CLASSES:
            indices = [i for i, r in enumerate(extra) if r['label'] == label]
            pool = [r for r in rows if r['label'] == label and r['preserve']]
            if len(pool) < len(indices):
                raise ValueError('Insufficient same-class controls')
            for i, j in zip(indices, rng.choice(len(pool), len(indices), replace=False), strict=True):
                picked[i] = pool[int(j)]
    return [dict(r, preserve=False, view='original') for _ in range(2) for r in picked]


def inputs(mode):
    rows, labels, ordinary, tensors = base_inputs('admit')
    if len(rows) != 1786 or int(ordinary.sum()) != 1602:
        raise ValueError('Shared blue-batch curriculum changed')
    extra = checked_extra()
    added = choose(rows, extra, mode)
    tensors[0] = np.concatenate((tensors[0], np.stack([load_crop(r)[0] for r in added])))
    tensors.append(np.stack([load_crop(r)[0] for r in extra]))
    return (rows + added, np.concatenate((labels, [CLASSES.index(r['label']) for r in added])),
            np.concatenate((ordinary, np.zeros(len(added), bool))), tensors)
