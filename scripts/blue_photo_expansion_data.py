"""Seven new-photo blue positives versus equal-count existing blue controls."""
import numpy as np
from scripts.blue_view_recovery_data import inputs as base_inputs
from scripts.blue_balloon_candidate_data import checked_extra
from scripts.grid_clutter_data import load_crop

def inputs(mode):
    if mode not in ('control','admit'):
        raise ValueError('Require paired arm')
    rows,labels,ordinary,tensors=base_inputs('admit')
    if len(rows)!=1750 or ordinary.sum()!=1602:
        raise ValueError('Frozen base changed')
    extra=checked_extra()
    if len(extra)!=7 or any(r['sha256'] in {old['sha256'] for old in rows} for r in extra):
        raise ValueError('Require seven novel reviewed crops')
    pool=[r for r in rows if r['label']=='blue_balloon' and r['preserve']]
    ids=np.random.default_rng(47).choice(len(pool),7,replace=False)
    chosen=extra if mode=='admit' else [pool[int(i)] for i in ids]
    added=[dict(r,preserve=False,view='original') for _ in range(2) for r in chosen]
    tensors=list(tensors)
    tensors[0]=np.concatenate((tensors[0],np.stack([load_crop(r)[0] for r in added])))
    tensors.append(np.stack([load_crop(r)[0] for r in extra]))
    return rows+added,np.concatenate((labels,np.full(14,2))),np.concatenate((ordinary,np.zeros(14,bool))),tensors
