"""Matched easy/hard TRAIN crop substitution with sealed visual-review provenance."""
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from dtr.data import read_json, sha256
from dtr.tracking import iou
from scripts.audit_blue_balloon_batch5 import checked_extra, REVIEW, OUTPUT as ADMISSION
from scripts.audit_batch5_scene_search import REVIEW as SCENE_REVIEW, OUTPUT as SCENES
from scripts.clutter_round3_feature_data import tensors as base_tensors
from scripts.grid_clutter_data import load_crop
from scripts.prepare_openimages_balloon_review import checked_path

PROTOCOL = Path('configs/batch5-occlusion-trial-20261011.json')
HARD_REVIEW = Path('configs/blue-balloon-batch5-hard-review-20261011.json')


def choose(rows, hard, arm, protocol):
    if arm not in protocol['arms'] or [r['id'] for r in rows] != protocol['easy_ids']:
        raise ValueError('Undeclared arm or easy inventory')
    old = next(r for r in rows if r['id'] == protocol['replace_easy_id'])
    if (hard['id'] != protocol['hard_id'] or hard['source_image'] != old['source_image']
            or hard['frame_id'] != old['frame_id'] or hard['sha256'] in {r['sha256'] for r in rows}
            or any(r['label'] != 'blue_balloon' or r['split'] != 'train'
                   or 'IMG_' in r['source_image'] for r in rows+[hard])):
        raise ValueError('Invalid same-photo reviewed substitution')
    result = [hard if arm == 'hard' and r['id'] == old['id'] else r for r in rows]
    counts = Counter(r['source_image'] for r in result)
    if len(result) != 7 or len(counts) != 5 or max(counts.values()) > 2:
        raise ValueError('Source balance changed')
    return result


def checked_batch():
    p = read_json(PROTOCOL)
    if (sha256(HARD_REVIEW) != p['hard_review_sha256']
            or sha256(SCENE_REVIEW) != p['scene_review_sha256']):
        raise ValueError('Sealed label review changed')
    easy = checked_extra()  # Rechecks licenses, original pixels and duplicate boundaries.
    review = read_json(HARD_REVIEW)
    root = Path(review['queue'])
    if (review['training_approved'] is not True or review['split'] != 'train'
            or review['test_evaluated'] is not False or review['deployment_approved'] is not False
            or review['labels'] != {'0': 'blue_balloon'}
            or sha256(root/'review.json') != review['queue_sha256']):
        raise ValueError('Hard crop not explicitly approved')
    queue, receipt = read_json(root/'review.json'), read_json(SCENES)
    if (queue['source_sha256'] != sha256(SCENES)
            or queue['script_sha256'] != sha256('scripts/summarize_batch5_scene_errors.py')
            or receipt['admission_sha256'] != sha256(ADMISSION)
            or sha256(receipt['raw']['path']) != receipt['raw']['sha256']
            or any(sha256(path) != h for path, h in receipt['inputs_sha256'].items())
            or len(queue['samples']) != 1):
        raise ValueError('Mining provenance changed')
    row = queue['samples'][0]
    scene_review = read_json(SCENE_REVIEW)
    source_root = Path(scene_review['queue'])
    if sha256(source_root/'review.json') != scene_review['queue_sha256']:
        raise ValueError('Scene queue changed')
    originals = read_json(source_root/'review.json')
    frame = originals['frames'][row['frame_id']]
    raw = read_json(receipt['raw']['path'])
    found = next(f for f in raw['frames'] if f['id'] == row['frame_id'])
    if (row['source'] != found['source'] or row['frame_sha256'] != found['frame_sha256']
            or frame['sha256'] != row['frame_sha256'] or row['target'] not in found['truth']
            or row['proposal'] not in found['detections'] or row['target']['annotation_id'] != 54
            or row['target']['label'] != 'blue_balloon'
            or iou(row['target']['box'], row['proposal']['box']) < .5
            or row['sha256'] != review['crop_sha256']['0']):
        raise ValueError('Hard crop is not the reviewed localized TRAIN target')
    path = checked_path(root, row['path'], row['sha256'])
    with Image.open(checked_path(source_root, frame['path'], frame['sha256'])) as im:
        pixels = np.asarray(im.convert('RGB'))
    a, b, c, d = row['proposal']['crop_box']
    if not (0 <= a < c <= 320 and 0 <= b < d <= 240):
        raise ValueError('Invalid crop bounds')
    with Image.open(path) as im:
        np.testing.assert_array_equal(np.asarray(im.convert('RGB')), pixels[b:d, a:c])
    hard = dict(row, source_image=row['source'], source_sha256=frame['original_sha256'],
                file=str(path), split='train', label='blue_balloon', preserve=False, view='original',
                crop_box=row['proposal']['crop_box'], attribution=frame['attribution'])
    paths = [PROTOCOL, HARD_REVIEW, REVIEW, SCENE_REVIEW, ADMISSION, SCENES,
             root/'review.json', path, source_root/'review.json', receipt['raw']['path'],
             __file__, 'scripts/audit_blue_balloon_batch5.py', 'scripts/reviewed_balloon_batch.py',
             'scripts/summarize_batch5_scene_errors.py']
    sources = {str(path): sha256(path) for path in paths}
    sources.update(receipt['inputs_sha256'])
    sources.update(read_json(ADMISSION)['inputs_sha256'])
    for r in easy:
        sources[r['file']] = r['sha256']
    return easy, hard, raw['frames'], sources


def tensors(arm):
    protocol = read_json(PROTOCOL)
    data = base_tensors(protocol['base_arm'])
    easy, hard, frames, sources = checked_batch()
    selected = choose(easy, hard, arm, protocol)
    if ({r['source_image'] for r in selected} & {r['source_image'] for r in data['rows']}
            or {r['sha256'] for r in selected} & {r['sha256'] for r in data['rows']}):
        raise ValueError('New batch overlaps old curriculum')
    added = selected * protocol['repetitions']
    pixels = np.stack([load_crop(r)[0] for r in added])
    n = len(added)
    for key, values in dict(x=pixels, y=np.full(n, 2), preserve=np.zeros(n, bool),
                            scene_mask=np.zeros(n, bool), weights=np.full(n, 4., np.float32)).items():
        data[key] = np.concatenate([data[key][:2126], values, data[key][2126:]])
    data['sample_identity'] += [{k:r[k] for k in ('source_image', 'sha256', 'label', 'preserve', 'view')}
                                for r in added]
    data['sources_sha256'].update(sources)
    data['p'] = dict(data['p'], arms=protocol['arms'], rows=protocol['rows'],
                     optimizer_steps=protocol['optimizer_steps'], gate=protocol['gate'], recipe=protocol['recipe'])
    if len(data['x']) != 2539 or n != 28 or data['scene_mask'].sum() != 385:
        raise ValueError('Curriculum dimensions changed')
    data.update(protocol=protocol, batch_rows=easy+[hard], batch_frames=frames,
                batch_pixels=np.stack([load_crop(r)[0] for r in easy+[hard]]))
    return data


def new_scene_gate(result, hard_correct):
    expected = {'red_balloon': (0, 0, 0), 'blue_balloon': (8, 0, 0)}
    return bool(hard_correct and all(tuple(result['metrics'][label][k] for k in ('tp', 'fp', 'fn')) == counts
                                    for label, counts in expected.items()))
