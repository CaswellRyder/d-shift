"""Verify visually reviewed TRAIN runtime crops and matched same-photo replacements."""
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, write_json
from dtr.tracking import iou
from scripts.audit_blue_balloon_window1 import checked_extra, OUTPUT as ADMISSION
from scripts.audit_window1_scene_search import OUTPUT as SCENES, REVIEW as SCENE_REVIEW
from scripts.prepare_openimages_balloon_review import checked_path
from scripts.summarize_public_balloon_trial import checked_metrics

REVIEW = Path('configs/blue-balloon-window1-hard-review-20261011.json')
REVIEW_SHA = '0f8c35de5ba78b1608d60ca654f3b764136db314ad0383312c2db4e4d72b4aad'
OUTPUT = Path('research/balloon-red-blue-20261008/window1-runtime-admission-20261011.json')
REPLACEMENTS = {0: 23, 1: 50}
EASY_IDS = [17, 18, 23, 25, 26, 34, 50, 53, 65]


def choose(easy, hard, arm):
    if (arm not in ('control', 'hard') or [r['id'] for r in easy] != EASY_IDS
            or [r['id'] for r in hard] != list(REPLACEMENTS)):
        raise ValueError('Undeclared arm or crop inventory')
    for r in easy + hard:
        if (r['split'] != 'train' or 'IMG_' in r['source_image']
                or not r['source_image'].startswith('openimages/train/')
                or r['label'] not in ('red_balloon', 'blue_balloon')):
            raise ValueError('Only explicitly reviewed TRAIN positives allowed')
    substitution = {}
    for r in hard:
        old = next(e for e in easy if e['id'] == REPLACEMENTS[r['id']])
        if (r['label'] != 'blue_balloon' or old['label'] != r['label']
                or any(r[k] != old[k] for k in ('source_image', 'source_sha256', 'frame_id'))
                or r['sha256'] in {e['sha256'] for e in easy}):
            raise ValueError('Runtime replacement changed target source or label')
        substitution[old['id']] = r
    result = [substitution.get(r['id'], r) if arm == 'hard' else r for r in easy]
    counts = Counter(r['source_image'] for r in result)
    if (len(result) != 9 or len(counts) != 6 or max(counts.values()) > 2
            or Counter(r['label'] for r in result) != {'blue_balloon': 6, 'red_balloon': 3}
            or len({r['sha256'] for r in result}) != 9):
        raise ValueError('Matched source balance changed')
    return result


def checked_batch():
    if sha256(REVIEW) != REVIEW_SHA:
        raise ValueError('Sealed runtime labels changed')
    review = read_json(REVIEW)
    if (review['split'] != 'train' or review['training_approved'] is not True
            or review['labels'] != {'0': 'blue_balloon', '1': 'blue_balloon'}
            or review['exclude'] != [2] or review['test_evaluated'] is not False
            or review['deployment_approved'] is not False
            or review['replace_easy_ids'] != {str(k): v for k, v in REPLACEMENTS.items()}):
        raise ValueError('Unapproved runtime label partition')
    easy = checked_extra()  # Rechecks original pixels, licenses and duplicate screening.
    queue_root = Path(review['queue'])
    if sha256(queue_root/'review.json') != review['queue_sha256']:
        raise ValueError('Runtime review queue changed')
    queue, scenes, admission = read_json(queue_root/'review.json'), read_json(SCENES), read_json(ADMISSION)
    if (len(queue['samples']) != 3 or [r['id'] for r in queue['samples']] != [0, 1, 2]
            or queue['source_sha256'] != sha256(SCENES)
            or queue['script_sha256'] != sha256('scripts/summarize_window1_scene_errors.py')
            or scenes['admission_sha256'] != sha256(ADMISSION)
            or sha256(scenes['raw']['path']) != scenes['raw']['sha256']):
        raise ValueError('Runtime mining provenance changed')
    sources = dict(admission['inputs_sha256'])
    sources.update(scenes['inputs_sha256'])
    if any(sha256(p) != digest for p, digest in sources.items()):
        raise ValueError('Upstream source receipt changed')
    raw = read_json(scenes['raw']['path'])
    if checked_metrics(raw) != scenes['metrics']:
        raise ValueError('Source scene counts do not reproduce')
    scene_review = read_json(SCENE_REVIEW)
    source_root = Path(scene_review['queue'])
    if sha256(source_root/'review.json') != scene_review['queue_sha256']:
        raise ValueError('Source review queue changed')
    originals = read_json(source_root/'review.json')
    hard = []
    for i, target_id in REPLACEMENTS.items():
        row = queue['samples'][i]
        frame = originals['frames'][row['frame_id']]
        found = next(f for f in raw['frames'] if f['id'] == row['frame_id'])
        if (row['source'] != found['source'] or row['frame_sha256'] != found['frame_sha256']
                or frame['sha256'] != row['frame_sha256']
                or row['source'] != 'openimages/train/'+frame['image_id']
                or row['target'] not in found['truth'] or row['proposal'] not in found['detections']
                or row['target']['annotation_id'] != target_id
                or row['target']['label'] != review['labels'][str(i)]
                or iou(row['target']['box'], row['proposal']['box']) < .5
                or row['sha256'] != review['crop_sha256'][str(i)]):
            raise ValueError('Runtime crop is not the reviewed localized TRAIN target')
        path = checked_path(queue_root, row['path'], row['sha256'])
        with Image.open(checked_path(source_root, frame['path'], frame['sha256'])) as im:
            rgb = np.asarray(im.convert('RGB'))
        a, b, c, d = row['proposal']['crop_box']
        if (not all(type(v) is int for v in (a, b, c, d))
                or not (0 <= a < c <= 320 and 0 <= b < d <= 240)):
            raise ValueError('Invalid native runtime crop bounds')
        with Image.open(path) as im:
            np.testing.assert_array_equal(np.asarray(im.convert('RGB')), rgb[b:d, a:c])
        hard.append(dict(row, source_image=row['source'], source_sha256=frame['original_sha256'],
            file=str(path), split='train', label=review['labels'][str(i)], preserve=False,
            view='original', crop_box=row['proposal']['crop_box'], attribution=frame['attribution']))
        sources[str(path)] = row['sha256']
    for arm in ('control', 'hard'):
        choose(easy, hard, arm)
    paths = [REVIEW, ADMISSION, SCENES, SCENE_REVIEW, queue_root/'review.json',
             source_root/'review.json', scenes['raw']['path'], __file__,
             'scripts/audit_blue_balloon_window1.py', 'scripts/reviewed_balloon_window.py',
             'scripts/summarize_window1_scene_errors.py', 'scripts/summarize_public_balloon_trial.py']
    sources.update({str(p): sha256(p) for p in paths})
    sources.update({r['file']: r['sha256'] for r in easy})
    return easy, hard, raw['frames'], sources


def audit():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    easy, hard, frames, sources = checked_batch()
    arms = {a: choose(easy, hard, a) for a in ('control', 'hard')}
    keys = ('id', 'source_image', 'source_sha256', 'sha256', 'label', 'file')
    write_json(OUTPUT, dict(arms={a: [{k: r[k] for k in keys} for r in rows] for a, rows in arms.items()},
        inputs_sha256=sources, changed_positions=[2, 6], runtime_crops_approved=2, excluded_runtime_ids=[2],
        crops_per_arm=9, source_photos_per_arm=6, label_counts={'blue_balloon': 6, 'red_balloon': 3},
        source_scene_metrics=checked_metrics(dict(frames=frames, metrics=read_json(SCENES)['metrics'])),
        scope='Verified same-photo TRAIN substitutions only. No model fitting or accuracy improvement demonstrated. Both arms retain nine crops from six photos; runtime views are not independent sources.',
        training_performed=False, development_evaluated=False, test_evaluated=False,
        pi_tested=False, deployment_approved=False, qualification_achieved=False))
    print(dict(approved=2, excluded=1, crops_per_arm=9, photos_per_arm=6), flush=True)


if __name__ == '__main__':
    audit()
