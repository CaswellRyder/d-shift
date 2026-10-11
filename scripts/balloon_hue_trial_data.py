"""Reconstruct reviewed TRAIN hue interventions and identical-budget controls."""
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from dtr.data import read_json, sha256
from scripts.window1_trial_data import tensors as base_tensors
from scripts.balloon_hue_augmentation import recolor, select

PROTOCOL = Path('configs/balloon-hue-trial-20261011.json')
REVIEW = Path('configs/balloon-hue-review-20261011.json')
START, STOP = 2162, 2290


def tensors(arm):
    protocol, review = read_json(PROTOCOL), read_json(REVIEW)
    if arm not in protocol['arms'] or review['training_approved'] is not True:
        raise ValueError('Undeclared arm or unapproved augmentation')
    root = Path(review['queue'])
    queue = read_json(root/'review.json')
    if sha256(root/'review.json') != review['queue_sha256']:
        raise ValueError('Changed hue queue')
    for p, h in {**queue['sources_sha256'], **queue['helpers_sha256']}.items():
        if sha256(p) != h:
            raise ValueError('Changed augmentation source')
    for f, h in review['sheets_sha256'].items():
        if sha256(root/f) != h or queue['sheets'][f] != h:
            raise ValueError('Changed reviewed pixels')
    data = base_tensors('hard')
    if select(data) != queue['selected_indexes']:
        raise ValueError('Source selection changed')
    samples = [r for r in queue['samples'] if r['id'] in review['approved_ids']]
    if len(samples) != 32 or len({r['id'] for r in samples}) != 32:
        raise ValueError('Review inventory changed')
    pixels, labels, identities = [], [], []
    for r in samples:
        i = r['original_index']
        row = data['rows'][i]
        changed, label, mask = recolor(data['x'][i], int(data['y'][i]))
        if (row['source_image'] != r['source'] or row['sha256'] != r['source_crop_sha256']
                or int(data['y'][i]) != r['label_before'] or label != r['label_after']
                or float(mask.mean()) != r['changed_fraction'] or sha256(root/r['path']) != r['sha256']):
            raise ValueError('Reviewed source identity changed')
        expected = np.concatenate([data['x'][i], changed], axis=1).astype(np.uint8)
        with Image.open(root/r['path']) as im:
            np.testing.assert_array_equal(np.asarray(im.convert('RGB')), expected)
        x = changed if arm == 'hue' else data['x'][i]
        y = label if arm == 'hue' else int(data['y'][i])
        pixels.append(x)
        labels.append(y)
        identities.append(dict(source_image=row['source_image'], sha256=row['sha256'],
            label=('background','red_balloon','blue_balloon')[y], preserve=False,
            view='reviewed_hue' if arm == 'hue' else 'original_replay',
            rendered_tensor_sha256=hashlib.sha256(x.tobytes()).hexdigest()))
    if labels.count(1) != 16 or labels.count(2) != 16:
        raise ValueError('Class balance changed')
    n = len(labels)*protocol['repetitions']
    values = dict(x=np.stack(pixels*protocol['repetitions']), y=np.array(labels*protocol['repetitions']),
        preserve=np.zeros(n, bool), scene_mask=np.zeros(n, bool), weights=np.full(n, 4., np.float32))
    for k, v in values.items():
        data[k] = np.concatenate([data[k][:START], v, data[k][START:]])
    data['sample_identity'] += identities*protocol['repetitions']
    data['sources_sha256'].update(queue['helpers_sha256'])
    paths = [PROTOCOL, REVIEW, root/'review.json', __file__]
    paths += [root/f for f in review['sheets_sha256']]
    paths += [root/r['path'] for r in samples]
    data['sources_sha256'].update({str(p):sha256(p) for p in paths})
    data['p'] = dict(data['p'], arms=protocol['arms'], rows=protocol['rows'],
        optimizer_steps=protocol['optimizer_steps'], recipe=protocol['recipe'], gate=protocol['gate'])
    data['protocol'] = protocol
    if (n != STOP-START or len(data['x']) != protocol['rows'] or data['scene_mask'].sum() != 385
            or int(np.ceil(len(data['x'])/data['p']['batch_size']))*data['p']['epochs'] != protocol['optimizer_steps']):
        raise ValueError('Augmentation budget changed')
    return data


def matched_inputs(a, b):
    for key in ('preserve','scene_mask','weights','probes','probe_labels','batch_pixels','batch_labels'):
        np.testing.assert_array_equal(a[key], b[key])
    for key in ('x', 'y'):
        np.testing.assert_array_equal(a[key][:START], b[key][:START])
        np.testing.assert_array_equal(a[key][STOP:], b[key][STOP:])
    np.testing.assert_array_equal(b['y'][START:STOP], 3-a['y'][START:STOP])
    if a['p'] != b['p'] or a['scene_ids'] != b['scene_ids']:
        raise ValueError('Unequal schedules or scene constraints')
    for i in range(START, STOP):
        expected, _, _ = recolor(a['x'][i], int(a['y'][i]))
        np.testing.assert_array_equal(b['x'][i], expected)
    return list(range(START, STOP))
