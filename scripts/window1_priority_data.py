"""TRAIN-derived positive weighting with exact frozen weight4 controls."""
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256
from scripts.window1_trial_data import tensors as previous_tensors
from scripts.diagnose_window1_objective import OUTPUT as DIAGNOSTIC

PROTOCOL = Path('configs/window1-priority-trial-20261011.json')


def required_multiplier(results):
    if not results:
        raise ValueError('Missing gradient evidence')
    required = 0.
    for row in results.values():
        new = np.asarray(row['gradients']['new_positive']['negative_gradient_margin_derivatives'], np.float64)
        total = np.asarray(row['gradients']['total']['negative_gradient_margin_derivatives'], np.float64)
        if new.shape != (2,) or total.shape != (2,) or not np.isfinite([new, total]).all() or np.any(new <= 0):
            raise ValueError('Need finite positive TRAIN margin response')
        required = max(required, float(np.max(1-total/new)))
    # Strictly exceed the worst local cancellation, choosing one power of two.
    return 1 if required < 1 else 2**(int(np.floor(np.log2(required)))+1)


def tensors(arm):
    if arm not in ('control', 'priority'):
        raise ValueError('Undeclared intervention')
    data = previous_tensors('hard')
    if arm == 'control':
        return data  # Exact completed hard47/hard48 trials are the matched controls.
    p, diagnostic = read_json(PROTOCOL), read_json(DIAGNOSTIC)
    if (sha256(DIAGNOSTIC) != p['diagnostic_sha256']
            or any(sha256(path) != h for path, h in diagnostic['inputs_sha256'].items())
            or required_multiplier(diagnostic['results']) != p['multiplier']
            or p['multiplier'] != 16 or p['positive_weight'] != 64):
        raise ValueError('TRAIN-derived weighting rule changed')
    if (not np.all(data['weights'][2126:2162] == 4)
            or np.any(data['scene_mask'][2126:2162]) or np.any(data['preserve'][2126:2162])):
        raise ValueError('Expected36new supervised rows only')
    data['weights'][2126:2162] = p['positive_weight']
    data['p'] = dict(data['p'], arms=['control', 'priority'], recipe=p['recipe'], gate=p['gate'])
    data['protocol'] = p
    data['sources_sha256'].update({str(path):sha256(path) for path in (PROTOCOL, DIAGNOSTIC, __file__)})
    return data


def matched_inputs(control, priority):
    for key in ('x', 'y', 'preserve', 'scene_mask', 'probes', 'probe_labels', 'batch_pixels', 'batch_labels'):
        np.testing.assert_array_equal(control[key], priority[key])
    changed = np.flatnonzero(control['weights'] != priority['weights']).tolist()
    if (changed != list(range(2126, 2162)) or not np.all(priority['weights'][changed] == 64)
            or not np.all(control['weights'][changed] == 4)
            or control['sample_identity'] != priority['sample_identity']
            or control['scene_ids'] != priority['scene_ids']):
        raise ValueError('Intervention differs beyond36positive weights')
    for key in control['p']:
        if key not in ('arms', 'recipe', 'gate') and control['p'][key] != priority['p'][key]:
            raise ValueError('Training schedule or model changed')
    return changed
