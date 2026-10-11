"""Build equal-budget mixed-color window1 TRAIN interventions."""
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256
from scripts.clutter_round3_feature_data import tensors as base_tensors
from scripts.grid_clutter_data import load_crop
from scripts.window1_runtime_data import checked_batch, choose, OUTPUT as ADMISSION
from scripts.train_batch3_retraining import LABELS

PROTOCOL = Path('configs/window1-runtime-trial-20261011.json')
CHANGED = [2128, 2132, 2137, 2141, 2146, 2150, 2155, 2159]


def tensors(arm):
    protocol = read_json(PROTOCOL)
    if arm not in protocol['arms']:
        raise ValueError('Undeclared arm')
    data = base_tensors(protocol['base_arm'])
    easy, hard, frames, sources = checked_batch()
    admission = read_json(ADMISSION)
    if any(sha256(p) != h for p, h in admission['inputs_sha256'].items()):
        raise ValueError('Runtime admission changed')
    selected = choose(easy, hard, arm)
    if ({r['source_image'] for r in selected} & {r['source_image'] for r in data['rows']}
            or {r['sha256'] for r in selected} & {r['sha256'] for r in data['rows']}):
        raise ValueError('Window1 overlaps old curriculum')
    added = selected * protocol['repetitions']
    n = len(added)
    values = dict(x=np.stack([load_crop(r)[0] for r in added]),
                  y=np.array([LABELS.index(r['label']) for r in added]),
                  preserve=np.zeros(n, bool), scene_mask=np.zeros(n, bool),
                  weights=np.full(n, data['p']['hard_weight'], np.float32))
    for key, value in values.items():
        data[key] = np.concatenate([data[key][:2126], value, data[key][2126:]])
    data['sample_identity'] += [{k:r[k] for k in ('source_image', 'sha256', 'label', 'preserve', 'view')}
                                for r in added]
    data['sources_sha256'].update(sources)
    data['sources_sha256'].update({str(p):sha256(p) for p in (PROTOCOL, ADMISSION, __file__)})
    data['p'] = dict(data['p'], arms=protocol['arms'], rows=protocol['rows'],
                    optimizer_steps=protocol['optimizer_steps'], recipe=protocol['recipe'], gate=protocol['gate'])
    if (len(data['x']) != 2547 or n != 36 or data['scene_mask'].sum() != 385
            or int(np.ceil(len(data['x'])/data['p']['batch_size']))*data['p']['epochs'] != 12800):
        raise ValueError('Curriculum or schedule changed')
    data.update(protocol=protocol, batch_rows=easy+hard, batch_frames=frames,
                batch_pixels=np.stack([load_crop(r)[0] for r in easy+hard]),
                batch_labels=np.array([LABELS.index(r['label']) for r in easy+hard]))
    return data


def matched_inputs(a, b):
    for key in ('y', 'preserve', 'scene_mask', 'weights', 'probes', 'probe_labels', 'batch_labels'):
        np.testing.assert_array_equal(a[key], b[key])
    changed = np.flatnonzero(np.any(a['x'] != b['x'], axis=(1, 2, 3))).tolist()
    if changed != CHANGED or a['p'] != b['p'] or a['scene_ids'] != b['scene_ids']:
        raise ValueError('Arms differ beyond eight repeated same-target replacements')
    return changed


def recovered_targets(result):
    from scripts.audit_current_blue_search import matched_truth
    expected = {9:23, 20:50}
    recovered = set()
    for frame in result['frames']:
        if frame['id'] in expected:
            for i in matched_truth(frame):
                target = frame['truth'][i]
                if target['label'] == 'blue_balloon' and target['annotation_id'] == expected[frame['id']]:
                    recovered.add(frame['id'])
    return recovered == set(expected)
