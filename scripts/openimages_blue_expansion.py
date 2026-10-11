"""Explicit blue crop review with pixel, license and TRAIN provenance checks."""
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, write_json
from scripts.prepare_blue_source_review import crop_box
from scripts.prepare_openimages_balloon_review import checked_path, sheets
from scripts.grid_clutter_data import data as base_data

REVIEW = Path('configs/openimages-blue-expansion2-20261011.json')
ROOT = Path('data/raw/openimages-balloon-expansion2-20261011')
PREVIEW = Path('data/openimages-blue-expansion2-selected-20261011')


def selection(review, samples, frames):
    if (review.get('split') != 'train' or review.get('label') != 'blue_balloon'
            or review.get('deployment_approved') is not False):
        raise ValueError('Require explicit TRAIN-only blue selection')
    ids = review['selected_crop_ids']
    if (any(type(i) is not int or not 0 <= i < len(samples) for i in ids)
            or len(ids) != len(set(ids))):
        raise ValueError('Invalid or repeated crop IDs')
    result = []
    for i in ids:
        row = samples[i]
        if (row['id'] != i or row['split'] != 'train'
                or row['source_image'] != 'openimages/train/' + frames[row['frame_id']]['image_id']):
            raise ValueError('Unexpected crop identity')
        result.append(dict(frame_id=row['frame_id'], box=row['box'], crop_box=row['crop_box'], queue_id=i))
    for row in review['manual_boxes']:
        f, box = row['frame_id'], row['box']
        if (type(f) is not int or not 0 <= f < len(frames) or len(box) != 4
                or any(type(v) is not int for v in box)
                or not (0 <= box[0] < box[2] <= 320 and 0 <= box[1] < box[3] <= 240)
                or min(box[2] - box[0], box[3] - box[1]) < 4):
            raise ValueError('Invalid manual box')
        result.append(dict(frame_id=f, box=box, crop_box=crop_box(box), queue_id=None))
    keys = [(r['frame_id'], tuple(r['crop_box'])) for r in result]
    if not result or len(keys) != len(set(keys)):
        raise ValueError('Empty or duplicate selected crops')
    return result


def checked_pixels():
    review = read_json(REVIEW)
    queue_root = Path(review['queue'])
    paths = {queue_root / 'review.json': review['queue_sha256'],
             ROOT / 'acquisition.json': review['acquisition_sha256'],
             ROOT / 'license-verification.json': review['license_sha256']}
    if any(sha256(p) != h for p, h in paths.items()):
        raise ValueError('Frozen source identity changed')
    queue, acquisition, licenses = [read_json(p) for p in paths]
    if (queue['acquisition_sha256'] != review['acquisition_sha256']
            or licenses['acquisition_sha256'] != review['acquisition_sha256']
            or acquisition['excluded_prior_acquisition_sha256'] != sha256('data/raw/openimages-balloon-train-20261011/acquisition.json')):
        raise ValueError('Acquisition lineage changed')
    for key, name in (('selection_sha256', 'selection.json'), ('attribution_sha256', 'attribution.json')):
        if acquisition[key] != sha256(ROOT / name):
            raise ValueError('Acquisition metadata changed')
    for path, digest in queue['reference_inputs'].items():
        if sha256(path) != digest:
            raise ValueError('Duplicate-screen reference changed')
    selected = selection(review, queue['samples'], queue['frames'])
    verified = {r['image_id'] for r in licenses['results'] if r['verified'] is True}
    sources = {r['image_id']: r for r in acquisition['images']}
    pixels = {}
    for f in sorted({r['frame_id'] for r in selected}):
        frame = queue['frames'][f]
        source = sources[frame['image_id']]
        if (frame['id'] != f or frame['image_id'] not in verified
                or frame['attribution']['Subset'] != 'train'
                or source['attribution'] != frame['attribution']
                or source['sha256'] != frame['original_sha256']
                or source['prior_train_hamming'] <= 8 or frame['reference_hamming'] <= 8
                or (frame['selected_hamming'] is not None and frame['selected_hamming'] <= 8)):
            raise ValueError('Unlicensed, duplicate or non-TRAIN selected frame')
        with Image.open(checked_path(ROOT, frame['original_path'], frame['original_sha256'])) as im:
            original = np.asarray(im.convert('RGB'))
        with Image.open(checked_path(queue_root, frame['path'], frame['sha256'])) as im:
            scan = np.asarray(im.convert('RGB'))
        if not np.array_equal(scan, cv2.resize(original, (320, 240), interpolation=cv2.INTER_AREA)):
            raise ValueError('Frame does not reproduce source')
        pixels[f] = scan
    output = []
    for i, r in enumerate(selected):
        a, b, c, d = r['crop_box']
        crop = pixels[r['frame_id']][b:d, a:c]
        if r['queue_id'] is not None:
            queued = queue['samples'][r['queue_id']]
            with Image.open(checked_path(queue_root, queued['path'], queued['sha256'])) as im:
                if not np.array_equal(crop, np.asarray(im.convert('RGB'))):
                    raise ValueError('Queued crop does not reproduce frame')
        frame = queue['frames'][r['frame_id']]
        output.append((dict(r, id=i, split='train', label='blue_balloon',
            source_image='openimages/train/' + frame['image_id'],
            source_sha256=frame['original_sha256'], attribution=frame['attribution']), crop))
    return output


def preview():
    if PREVIEW.exists():
        raise FileExistsError(PREVIEW)
    selected = checked_pixels()
    PREVIEW.mkdir(parents=True)
    rows = []
    for row, pixels in selected:
        path = f"{row['id']:03d}.png"
        Image.fromarray(pixels).save(PREVIEW / path)
        rows.append(dict(row, path=path, sha256=sha256(PREVIEW / path)))
    sheets(PREVIEW, 'crops', rows, 6, (160, 160))
    write_json(PREVIEW / 'preview.json', dict(samples=rows,
        queue_sha256=read_json(REVIEW)['queue_sha256'],
        sheets={p.name: sha256(p) for p in PREVIEW.glob('crops-*.jpg')},
        training_approved=False, deployment_approved=False))
    print(dict(crops=len(rows), scenes=len({r['frame_id'] for r in rows}),
        preview_sha256=sha256(PREVIEW / 'preview.json')))


def checked_extra():
    review, evidence = read_json(REVIEW), read_json(PREVIEW / 'preview.json')
    if (review.get('training_approved') is not True
            or review.get('preview_sha256') != sha256(PREVIEW / 'preview.json')
            or review.get('reviewed_sheets') != evidence['sheets']):
        raise ValueError('Final crop review is not approved')
    for name, digest in evidence['sheets'].items():
        checked_path(PREVIEW, name, digest)
    selected = checked_pixels()
    if len(selected) != len(evidence['samples']):
        raise ValueError('Selected count changed')
    result = []
    for (row, pixels), saved in zip(selected, evidence['samples'], strict=True):
        if any(saved[k] != v for k, v in row.items()):
            raise ValueError('Selected identity changed')
        path = checked_path(PREVIEW, saved['path'], saved['sha256'])
        with Image.open(path) as im:
            if not np.array_equal(pixels, np.asarray(im.convert('RGB'))):
                raise ValueError('Reviewed crop pixels changed')
        result.append(dict(saved, file=str(path), view='original', preserve=False))
    return result


def data(mode):
    if mode not in ('control', 'admit'):
        raise ValueError('Unknown paired arm')
    rows, clutter = base_data('admit')
    extra = checked_extra()
    known = {r['sha256'] for r in rows}
    if len({r['sha256'] for r in extra}) != len(extra) or any(r['sha256'] in known for r in extra):
        raise ValueError('Repeated TRAIN crop')
    pool = [r for r in rows if r['label'] == 'blue_balloon' and r['preserve']]
    indices = np.random.default_rng(47).choice(len(pool), len(extra), replace=False)
    selected = extra if mode == 'admit' else [pool[int(i)] for i in indices]
    additions = [dict(r, preserve=False, view='original') for _ in range(2) for r in selected]
    return rows + additions, extra, clutter


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview', action='store_true', required=True)
    parser.parse_args()
    preview()
