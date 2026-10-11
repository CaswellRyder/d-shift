"""Add only reviewed blue runtime views to the verified blue-object curriculum."""
import numpy as np

from scripts.blue_object_data import inputs as base_inputs
from scripts.runtime_positive_data import checked_extra
from scripts.audit_runtime_positive_coverage import digest
from scripts.grid_clutter_data import load_crop


def inputs(mode):
    if mode not in ('control','admit'):
        raise ValueError('Require paired arm')
    rows,labels,ordinary,tensors=base_inputs('admit')
    if len(rows)!=1681 or ordinary.sum()!=1533:
        raise ValueError('Unexpected frozen base')
    extra=[r for r in checked_extra({digest(x) for x in tensors[0]}) if r['label']=='blue_balloon']
    if len(extra)!=69:
        raise ValueError('Require69 previously reviewed blue views')
    pool=[r for r in rows if r['label']=='blue_balloon' and r['preserve']]
    ids=np.random.default_rng(47).choice(len(pool),len(extra),replace=False)
    chosen=extra if mode=='admit' else [pool[int(i)] for i in ids]
    added=[dict(r,preserve=True,view='original') for r in chosen]
    tensors[0]=np.concatenate((tensors[0],np.stack([load_crop(r)[0] for r in added])))
    tensors.append(np.stack([load_crop(r)[0] for r in extra]))
    return rows+added,np.concatenate((labels,np.full(69,2))),np.concatenate((ordinary,np.ones(69,bool))),tensors
