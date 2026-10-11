"""Verify exact reviewed runtime crops against screened TRAIN source pixels."""
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from dtr.data import read_json, sha256
from scripts.indoor_expansion5_data import checked_extra as old_rows, QUEUE as SOURCE
from scripts.openimages_balloon_training import review_labels
from scripts.prepare_indoor5_runtime_review import OUTPUT as QUEUE
from scripts.prepare_openimages_balloon_review import checked_path

REVIEW = Path('configs/indoor5-runtime-review-20261011.json')


def validate_labels(review, count):
    labels = review_labels(review, count)
    if Counter(labels.values()) != dict(background=2, red_balloon=33, blue_balloon=29):
        raise ValueError('Require explicitly reviewed64-crop partition')
    return labels


def checked_extra():
    old = old_rows()  # Includes archive reconstruction and whole-IMG duplicate guards.
    queue, review, source = map(read_json, (QUEUE / 'review.json', REVIEW, SOURCE / 'review.json'))
    if (review['queue'] != str(QUEUE) or review['queue_sha256'] != sha256(QUEUE / 'review.json')
            or review['reviewed_sheets'] != queue['sheets']
            or queue['script_sha256'] != sha256('scripts/prepare_indoor5_runtime_review.py')
            or queue['training_approved'] is not False or queue['test_evaluated'] is not False):
        raise ValueError('Runtime crop review changed')
    for path, digest in {**queue['inputs_sha256'], **queue['helpers_sha256']}.items():
        if sha256(path) != digest:
            raise ValueError('Source or mining helper changed')
    for path, identity in queue['models'].items():
        if sha256(path) != identity['sha256'] or sha256(Path(path).with_suffix('.json')) != identity['metadata_sha256']:
            raise ValueError('Mining model changed')
    for name, digest in queue['sheets'].items():
        checked_path(QUEUE, name, digest)
    labels = validate_labels(review, len(queue['samples']))
    eligible = {r['frame_id']: r for r in old}
    old_hashes = {r['sha256'] for r in old}
    images, result = {}, []
    for i, label in sorted(labels.items()):
        row = queue['samples'][i]
        frame = queue['frames'][row['frame_id']]
        source_id = frame['source_frame_id']
        original = source['frames'][source_id]
        if (source_id not in eligible or row['id'] != i or frame['id'] != row['frame_id']
                or row['source_frame_id'] != source_id or frame['source'] != original['source']
                or frame['source_frame_sha256'] != original['sha256']
                or not original['source'].startswith('frame_') or 'IMG_' in original['source']
                or row['source_image'] != 'train/' + original['source'] or row['split'] != 'train'
                or row['sha256'] in old_hashes):
            raise ValueError('Reserved, duplicate or changed source')
        if row['frame_id'] not in images:
            with Image.open(checked_path(QUEUE, frame['path'], frame['sha256'])) as image:
                rgb = np.asarray(image.convert('RGB'))
            with Image.open(checked_path(SOURCE, original['path'], original['sha256'])) as image:
                if not np.array_equal(rgb, np.asarray(image.convert('RGB'))):
                    raise ValueError('Runtime frame differs from verified source')
            images[row['frame_id']] = rgb
        rgb = images[row['frame_id']]
        a, b, c, d = row['crop_box']
        if not (0 <= a < c <= 320 and 0 <= b < d <= 240):
            raise ValueError('Invalid runtime crop bounds')
        path = checked_path(QUEUE, row['path'], row['sha256'])
        with Image.open(path) as image:
            if not np.array_equal(np.asarray(image.convert('RGB')), rgb[b:d, a:c]):
                raise ValueError('Runtime crop differs from source pixels')
        result.append(dict(row, label=label, file=str(path), preserve=False, view='original',
                           source_sha256=eligible[source_id]['source_sha256']))
    if len({r['sha256'] for r in result}) != len(result):
        raise ValueError('Duplicate reviewed crop pixels')
    return result
