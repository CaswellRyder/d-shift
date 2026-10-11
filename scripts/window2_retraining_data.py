"""Class-matched real window2 TRAIN additions, with frozen prior scene constraints."""
from collections import Counter
from pathlib import Path

import numpy as np

from dtr.data import read_json,sha256
from scripts.window1_trial_data import tensors as base_tensors
from scripts.audit_blue_balloon_window2 import checked_extra,OUTPUT as ADMISSION,REVIEW
from scripts.grid_clutter_data import load_crop
from scripts.train_batch3_retraining import LABELS

PROTOCOL = Path('configs/window2-retraining-20261011.json')
START,STOP = 2162,2222


def choose_controls(rows,extra):
    rng,selected,counts = np.random.default_rng(47),[],Counter()
    for label in LABELS:
        n = sum(r['label']==label for r in extra)
        pool = list({r['sha256']:r for r in rows if r['label']==label and r['preserve'] and r['view']=='original'}.values())
        for i in rng.permutation(len(pool)):
            if sum(r['label']==label for r in selected) == n:
                break
            row = pool[int(i)]
            if counts[row['source_image']] < 2:
                selected.append(row)
                counts[row['source_image']] += 1
        if sum(r['label']==label for r in selected) != n:
            raise ValueError('Insufficient unique source-capped controls')
    return selected


def tensors(arm):
    p = read_json(PROTOCOL)
    if arm not in p['arms']:
        raise ValueError('Unknown intervention')
    data = base_tensors('hard')
    extra = sorted(checked_extra(),key=lambda r:(LABELS.index(r['label']),r['id']))
    admission = read_json(ADMISSION)
    if any(sha256(path)!=h for path,h in admission['inputs_sha256'].items()):
        raise ValueError('Frozen window2 admission changed')
    if ({r['source_image'] for r in extra}&{r['source_image'] for r in data['rows']}
            or {r['sha256'] for r in extra}&{r['sha256'] for r in data['rows']}):
        raise ValueError('New sources overlap existing curriculum')
    chosen = extra if arm=='admit' else choose_controls(data['rows'],extra)
    added = [dict(r,preserve=False,view='original') for r in chosen]*p['repetitions']
    n = len(added)
    values = dict(x=np.stack([load_crop(r)[0] for r in added]),
        y=np.array([LABELS.index(r['label']) for r in added]),preserve=np.zeros(n,bool),
        scene_mask=np.zeros(n,bool),weights=np.full(n,4.,np.float32))
    for k,v in values.items():
        data[k] = np.concatenate([data[k][:START],v,data[k][START:]])
    data['sample_identity'] += [{k:r[k] for k in ('source_image','sha256','label','preserve','view')} for r in added]
    data['window2_pixels'] = np.stack([load_crop(r)[0] for r in extra])
    data['window2_labels'] = np.array([LABELS.index(r['label']) for r in extra])
    data['window2_ids'] = [r['id'] for r in extra]
    data['sources_sha256'].update(admission['inputs_sha256'])
    data['sources_sha256'].update({str(path):sha256(path) for path in (PROTOCOL,ADMISSION,REVIEW,__file__)})
    data['p'] = dict(data['p'],arms=p['arms'],rows=p['rows'],optimizer_steps=p['optimizer_steps'],recipe=p['recipe'],gate=p['gate'])
    data['protocol'] = p
    if (len(extra)!=15 or n!=STOP-START or len(data['x'])!=p['rows'] or data['scene_mask'].sum()!=385
            or int(np.ceil(len(data['x'])/data['p']['batch_size']))*data['p']['epochs']!=p['optimizer_steps']):
        raise ValueError('Training budget changed')
    return data


def matched_inputs(a,b):
    for k in ('y','preserve','scene_mask','weights','probes','probe_labels','batch_pixels','batch_labels','window2_pixels','window2_labels'):
        np.testing.assert_array_equal(a[k],b[k])
    np.testing.assert_array_equal(a['x'][:START],b['x'][:START])
    np.testing.assert_array_equal(a['x'][STOP:],b['x'][STOP:])
    for key in ('p','scene_ids','window2_ids'):
        if a[key]!=b[key]:
            raise ValueError('Schedule or probe identity changed')
    if a['sample_identity'][:START]!=b['sample_identity'][:START]:
        raise ValueError('Original supervised identity changed')
    return np.flatnonzero(np.any(a['x']!=b['x'],axis=(1,2,3))).tolist()
