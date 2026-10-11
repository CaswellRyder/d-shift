"""Matched runtime positives alongside shared reviewed blue-clutter negatives."""
from collections import Counter

import numpy as np

from scripts.blue_objects_round2_trial_data import inputs as base_inputs
from scripts.grid_clutter_data import load_crop
from scripts.indoor5_runtime_data import checked_extra

LABELS = ('background','red_balloon','blue_balloon')


def choose(rows, extra, mode):
    expected = dict(background=2,red_balloon=33,blue_balloon=29)
    if mode not in ('control','admit') or Counter(r['label'] for r in extra) != expected:
        raise ValueError('Require declared arm and reviewed64-crop partition')
    rng, picked = np.random.default_rng(47), []
    for label in LABELS:
        eligible = [r for r in extra if r['label'] == label]
        if mode == 'admit':
            picked.extend(eligible)
        else:
            pool = list({r['sha256']:r for r in rows if r['label'] == label and r['preserve']}.values())
            if len(pool) < len(eligible):
                raise ValueError('Insufficient distinct matched controls')
            picked.extend(pool[int(i)] for i in rng.choice(len(pool),len(eligible),replace=False))
    return [dict(r,preserve=False,view=view) for view in ('original','horizontal_flip') for r in picked]


def crop(row):
    # Existing loader does not know this new, explicitly declared augmentation.
    if row['view'] not in ('original','horizontal_flip'):
        raise ValueError('Undeclared runtime augmentation')
    values = load_crop(dict(row,view='original'))[0]
    return values.copy() if row['view'] == 'original' else values[:,::-1].copy()


def inputs(mode):
    rows, labels, ordinary, tensors = base_inputs('admit')
    extra = checked_extra()
    if len(rows) != 1946 or int(ordinary.sum()) != 1602 or len(tensors) != 10:
        raise ValueError('Shared curriculum changed')
    if {r['sha256'] for r in rows} & {r['sha256'] for r in extra}:
        raise ValueError('Runtime crops duplicate base pixels')
    added = choose(rows,extra,mode)
    if any(r['split'] != 'train' or 'IMG_' in r['source_image'] for r in added):
        raise ValueError('Unsafe training source')
    originals = np.stack([crop(dict(r,view='original')) for r in extra])
    tensors[0] = np.concatenate((tensors[0],np.stack([crop(r) for r in added])))
    tensors.extend((originals,originals[:,:,::-1].copy()))
    runtime_labels = np.array([LABELS.index(r['label']) for r in extra])
    return (rows+added,np.concatenate((labels,[LABELS.index(r['label']) for r in added])),
        np.concatenate((ordinary,np.zeros(128,bool))),tensors,runtime_labels)
