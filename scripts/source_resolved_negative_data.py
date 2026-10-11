"""Validated source-resolution background labels; identical-input matched training."""
from pathlib import Path

import numpy as np
from PIL import Image

from dtr.data import read_json, sha256
from scripts.prepare_source_resolved_negatives import ROOT, AUDIT, AUDIT_SHA
from scripts.prepare_direct_hybrid_negatives import clear_of_targets
from scripts.audit_scale_balanced_search import checked_panels
from scripts.positive_proposal_correction_data import tensors as previous_tensors, matched as compare
from scripts.public_scene_localizer_data import tensor_hash

PROTOCOL = Path('configs/source-resolved-negative-retraining-20261011.json')
REVIEW = Path('configs/source-resolved-negative-review-20261011.json')


def outside_boxes(box, truth):
    a, b, c, d = box
    if not 0 <= a < c <= 320 or not 0 <= b < d <= 240:
        raise ValueError('Invalid negative crop')
    return not any(min(c, t['box'][2]) > max(a, t['box'][0]) and
                   min(d, t['box'][3]) > max(b, t['box'][1]) for t in truth)


def assert_novel(base, added):
    prior = {tensor_hash(x) for x in base}
    hashes = [tensor_hash(x) for x in added]
    if len(set(hashes)) != len(hashes) or prior.intersection(hashes):
        raise ValueError('New negatives duplicate previous training inputs')


def checked_extra():
    review, q = read_json(REVIEW), read_json(ROOT/'review.json')
    if (review['queue'] != str(ROOT) or sha256(ROOT/'review.json') != review['queue_sha256']
            or review['training_approved'] is not True or review['assistant_visual_review'] is not True
            or review['human_review_verified'] is not False
            or any(review[k] is not False for k in ('development_evaluated', 'test_evaluated',
                'pi_tested', 'deployment_approved', 'qualification_achieved'))):
        raise ValueError('Reviewed scope or identity changed')
    if (sha256(AUDIT) != AUDIT_SHA or q['audit_sha256'] != AUDIT_SHA
            or sha256('scripts/prepare_source_resolved_negatives.py') != q['script_sha256']
            or sha256(ROOT/'sheet.png') != q['sheet_sha256']
            or sha256(read_json(AUDIT)['raw']['path']) != q['raw_sha256']):
        raise ValueError('Review evidence changed')
    if (set(review['labels']) != {'0', '1', '2'} or set(review['labels'].values()) != {'background'}
            or set(review['review_notes']) != set(review['labels'])
            or len(q['samples']) != 3 or {r['id'] for r in q['samples']} != {0, 1, 2}):
        raise ValueError('Require all three explicit background labels')
    panels = checked_panels()
    rows, images, exceptions = [], [], set()
    for r in q['samples']:
        if (r['split'] != 'train' or not r['source'].startswith('openimages/train/')
                or 'IMG_' in r['source'] or not outside_boxes(r['crop_box'], r['truth'])):
            raise ValueError('Unapproved or target-containing crop')
        guard = clear_of_targets(r['crop_box'], r['truth'])
        if guard != r['clear_of_16px_guard']:
            raise ValueError('Guard geometry changed')
        if not guard:
            exceptions.add(str(r['id']))
        for key in ('crop', 'original', 'detail'):
            if sha256(r[key+'_path']) != r[key+'_sha256']:
                raise ValueError('Reviewed pixels changed')
        frame, rgb = next((f, rgb) for f, rgb in panels[r['panel']] if f['source'] == r['source'])
        if any(frame[k] != r[k] for k in ('frame_sha256', 'truth')):
            raise ValueError('Source label identity changed')
        with Image.open(r['crop_path']) as im:
            native = np.asarray(im.convert('RGB'))
            pixel = np.asarray(im.convert('RGB').resize((64, 64), Image.Resampling.BILINEAR), np.float32)
        a, b, c, d = r['crop_box']
        np.testing.assert_array_equal(native, rgb[b:d, a:c])
        rows.append(dict(r, label='background', path=str(Path(r['crop_path']).resolve())))
        images.append(pixel)
    if exceptions != set(review['guard_exceptions']) or any(not x for x in review['guard_exceptions'].values()):
        raise ValueError('Every near-target crop needs an exact visual exception')
    return rows, np.stack(images)


def run_path(arm, seed):
    if arm not in ('control', 'corrected') or seed not in (47, 48):
        raise ValueError('Undeclared run')
    return Path(f'runs/source-resolved-negative-{arm}-seed{seed}-20261011')


def tensors(arm, seed):
    run_path(arm, seed)
    p = read_json(PROTOCOL)
    if (p['training_approved'] is not True or sha256(p['parent_audit']) != p['parent_audit_sha256']
            or p['review'] != str(REVIEW) or any(p[k] is not False for k in
                ('development_evaluated', 'test_evaluated', 'pi_tested', 'deployment_approved', 'qualification_achieved'))):
        raise ValueError('Training scope changed')
    parent = p['parents'][str(seed)]
    if sha256(parent['path']) != parent['sha256']:
        raise ValueError('Parent checkpoint changed')
    record = read_json(p['parent_audit'])['training'][f'corrected{seed}']
    for name, digest in record['artifacts_sha256'].items():
        if sha256(Path(record['run'])/name) != digest:
            raise ValueError('Parent artifacts changed')
    data = previous_tensors('corrected', seed)
    rows, pixels = checked_extra()
    if len(data['x']) != 2668 or len(rows) != p['new_negatives']:
        raise ValueError('Curriculum changed')
    assert_novel(data['x'], pixels)
    n = len(rows)*p['repetitions']
    positions = list(range(len(data['x']), len(data['x'])+n))
    corrected = arm == 'corrected'
    for key, values in dict(x=np.tile(pixels, (p['repetitions'], 1, 1, 1)),
            y=np.full(n, 0 if corrected else -1, int), scene_mask=np.full(n, not corrected, bool),
            preserve=np.zeros(n, bool), weights=np.full(n, p['hard_weight'] if corrected else 1, np.float32)).items():
        data[key] = np.concatenate([data[key], values])
    if len(data['x']) != p['rows'] or int(np.ceil(p['rows']/p['batch_size']))*p['epochs'] != p['optimizer_steps']:
        raise ValueError('Matched budget changed')
    paths = [PROTOCOL, REVIEW, ROOT/'review.json', __file__, p['parent_audit'],
             'scripts/prepare_source_resolved_negatives.py', 'scripts/train_source_resolved_negatives.py']
    data['sources_sha256'].update({str(path): sha256(path) for path in paths})
    data.update(settings=p, arm=arm, seed=seed, parent=parent, corrections=positions,
        reviewed_negatives=data['reviewed_negatives']+rows,
        negative_pixels=np.concatenate([data['negative_pixels'], pixels]))
    return data


def matched(a, b):
    compare(a, b)
