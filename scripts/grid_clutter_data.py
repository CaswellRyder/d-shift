"""Reviewed TRAIN backgrounds with an explicit synthetic red-surface-to-blue augmentation."""
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, write_json
from scripts.audit_balloon_search_stages import training_source
from scripts.balloon_dense_complete import validate_derivative
from scripts.prepare_grid_clutter_review import BASE, BASE_SHA, guarded_background
from scripts.train_balloon_dense_proposals import checked_data as dense_data
from scripts.train_balloon_fragments import checked_data, MANIFEST, QUEUE as FRAGMENT_QUEUE
from scripts.prepare_openimages_balloon_review import sheets

QUEUE = Path('data/balloon-grid-clutter-review-v2-20261011')
REVIEW = Path('configs/balloon-grid-clutter-review-20261011.json')


def blue_surface(rgb):
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError('Require RGB uint8 crop')
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = ((hsv[..., 0] <= 10) | (hsv[..., 0] >= 170)) & (hsv[..., 1] >= 80) & (hsv[..., 2] >= 40)
    hsv[..., 0][mask] = 110
    changed = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    result = rgb.copy()
    result[mask] = changed[mask]
    return result, float(mask.mean())


def checked_extra():
    review, queue = read_json(REVIEW), read_json(QUEUE / 'review.json')
    if (review['queue'] != str(QUEUE) or review['queue_sha256'] != sha256(QUEUE / 'review.json')
            or review.get('training_approved') is not True or review.get('split') != 'train'
            or review.get('deployment_approved') is not False
            or queue['base_manifest_sha256'] != BASE_SHA or sha256(BASE / 'manifest.json') != BASE_SHA
            or queue['script_sha256'] != sha256('scripts/prepare_grid_clutter_review.py')):
        raise ValueError('Reviewed source identity changed')
    ids = review['background_ids'] + [int(i) for i in review['exclude']]
    if len(ids) != len(set(ids)) or set(ids) != set(range(len(queue['samples']))):
        raise ValueError('Every crop needs exactly one explicit decision')
    if review['reviewed_sheets'] != queue['sheets']:
        raise ValueError('Reviewed sheets changed')
    for name, digest in queue['sheets'].items():
        if sha256(QUEUE / name) != digest:
            raise ValueError('Review sheet pixels changed')
    base, _, _, folds, _ = dense_data(BASE)
    complete = validate_derivative(base)
    for identity in queue['models'].values():
        if sha256(identity['path']) != identity['sha256'] or sha256(Path(identity['path']).with_suffix('.json')) != identity['metadata_sha256']:
            raise ValueError('Mining localizer identity changed')
    selected = []
    for i in review['background_ids']:
        row = queue['samples'][i]
        frame = queue['frames'][row['frame_id']]
        index = row['base_index']
        source = base['frames'][index]
        training_source(row['source'])
        if (index not in complete or folds[index] == 0 or row['id'] != i or row['split'] != 'train'
                or frame['base_index'] != index or frame['source'] != row['source']
                or row['source'] != source['source'] or row['source_sha256'] != source['source_sha256']
                or frame['truth'] != source['truth'] or not guarded_background(row['crop_box'], frame['truth'])):
            raise ValueError('Unsafe or unapproved background crop')
        frame_path = (QUEUE / frame['path']).resolve()
        crop_path = (QUEUE / row['path']).resolve()
        if (not frame_path.is_relative_to(QUEUE.resolve()) or not crop_path.is_relative_to(QUEUE.resolve())
                or sha256(frame_path) != frame['sha256'] or sha256(crop_path) != row['sha256']):
            raise ValueError('Changed or unsafe review pixels')
        with Image.open(BASE / source['path']) as image:
            original = np.asarray(image.convert('RGB'))
        with Image.open(frame_path) as image:
            frame_pixels = np.asarray(image.convert('RGB'))
        with Image.open(crop_path) as image:
            crop = np.asarray(image.convert('RGB'))
        a, b, c, d = row['crop_box']
        if not np.array_equal(original, frame_pixels) or not np.array_equal(crop, original[b:d, a:c]):
            raise ValueError('Crop does not reproduce original TRAIN pixels')
        selected.append(dict(row, label='background', source_image=row['source'], file=str(crop_path)))
    return selected


def data(mode):
    if mode not in ('control', 'admit'):
        raise ValueError('Require paired mode')
    ordinary, fragments = checked_data()
    extra = checked_extra()
    rows = [dict(r, file=str(Path(MANIFEST).parent / r['path']), preserve=True, view='original') for r in ordinary]
    rows += [dict(r, label='blue_balloon', file=str(Path(FRAGMENT_QUEUE) / r['path']), preserve=True,
                  view='original') for r in fragments] * 4
    known = {r['sha256']: r['label'] for r in rows}
    if any(r['sha256'] in known and known[r['sha256']] != 'background' for r in extra):
        raise ValueError('Negative conflicts with a known positive')
    pool = [r for r in rows if r['label'] == 'background']
    rng = np.random.default_rng(47)
    selected = extra if mode == 'admit' else [pool[int(i)] for i in rng.choice(len(pool), len(extra), replace=False)]
    additions = [dict(r, preserve=False, view=view) for _ in range(2) for r in selected for view in ('original', 'blue_surface')]
    return rows + additions, extra


def load_crop(row):
    with Image.open(row['file']) as image:
        pixels = np.asarray(image.convert('RGB'))
    fraction = 0.
    if row['view'] == 'blue_surface':
        pixels, fraction = blue_surface(pixels)
    elif row['view'] != 'original':
        raise ValueError('Unknown augmentation')
    # Match dtr.data.load_rgb: PIL bilinear resizing, floating input0..255.
    resized = Image.fromarray(pixels).resize((64, 64), Image.Resampling.BILINEAR)
    return np.asarray(resized, np.float32), fraction


def preview(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    extra = checked_extra()
    output.mkdir(parents=True)
    records = []
    for row in extra:
        with Image.open(row['file']) as image:
            augmented, fraction = blue_surface(np.asarray(image.convert('RGB')))
        path = f"{row['id']:03d}.png"
        Image.fromarray(augmented).save(output / path)
        records.append(dict(id=row['id'], frame_id=row['frame_id'], path=path,
            sha256=sha256(output / path), source_crop_sha256=row['sha256'], changed_fraction=fraction,
            label='background', synthetic=True, split='train'))
    sheets(output, 'crops', records, 6, (160, 160))
    write_json(output / 'augmentation.json', dict(samples=records, source_review_sha256=sha256(REVIEW),
        script_sha256=sha256(__file__), scope='Synthetic TRAIN augmentation of reviewed real backgrounds, not new real scenes',
        test_evaluated=False, deployment_approved=False))
    print(dict(samples=len(records), changed_fractions=[r['changed_fraction'] for r in records],
               receipt_sha256=sha256(output / 'augmentation.json')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview', required=True)
    preview(parser.parse_args().preview)
