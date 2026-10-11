"""Acquire a bounded, disjoint public TRAIN batch; no automatic color labels."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
from pathlib import Path

import numpy as np
from PIL import Image
import requests

from dtr.data import read_json, sha256, write_json
from scripts.acquire_openimages_balloons import (BALLOON, BOX_URL, CLASS_URL, IMAGE_ROOT,
                                                bounded_get, complete_boxes, metadata_for)
from scripts.prepare_indoor_balloon_review import fingerprint, holdout_distance

PRIOR = Path('data/raw/openimages-balloon-train-20261011')


def unseen_ids(groups, excluded, limit=96):
    if type(limit) is not int or not 1 <= limit <= 96:
        raise ValueError('Require bounded1..96 TRAIN images')
    return sorted(set(groups) - set(excluded))[:limit]


def download(identifier):
    url = IMAGE_ROOT + identifier + '.jpg'
    try:
        raw, headers = bounded_get(url, 12 << 20)
        with Image.open(io.BytesIO(raw)) as image:
            if image.format != 'JPEG' or image.width * image.height > 4_000_000:
                raise ValueError('Unexpected source format or dimensions')
            rgb = np.asarray(image.convert('RGB'))
        return dict(image_id=identifier, raw=raw, rgb=rgb, url=url, headers=headers)
    except (requests.RequestException, ValueError, OSError) as error:
        return dict(image_id=identifier, error=str(error))


def acquire(output, limit=96):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    prior, selection = read_json(PRIOR / 'acquisition.json'), read_json(PRIOR / 'selection.json')
    if prior['selection_sha256'] != sha256(PRIOR / 'selection.json'):
        raise ValueError('Prior selection changed')
    excluded = selection['selected_image_ids']
    prints = []
    for image in prior['images']:
        path = (PRIOR / image['path']).resolve()
        if not path.is_relative_to(PRIOR.resolve()) or sha256(path) != image['sha256']:
            raise ValueError('Prior image changed')
        with Image.open(path) as opened:
            prints.append(fingerprint(np.asarray(opened.convert('RGB'))))
    classes, _ = bounded_get(CLASS_URL, 1 << 20)
    if dict(csv.reader(io.StringIO(classes.decode()))).get(BALLOON) != 'Balloon':
        raise ValueError('Source class changed')
    scan_bytes = 64 << 20
    raw, headers = bounded_get(BOX_URL, scan_bytes, {'Range': f'bytes=0-{scan_bytes-1}'}, 206)
    digest = hashlib.sha256(raw).hexdigest()
    if (len(raw) != scan_bytes or digest != selection['scan_sha256']
            or not headers.get('Content-Range', '').startswith(f'bytes 0-{scan_bytes-1}/')):
        raise ValueError('Pinned annotation window changed')
    groups = complete_boxes(raw)
    selected = unseen_ids(groups, excluded, limit)
    if not selected:
        raise ValueError('No new TRAIN images in bounded window')
    output.mkdir(parents=True)
    write_json(output / 'selection.json', dict(annotations_url=BOX_URL, headers=headers,
        scan_bytes=scan_bytes, scan_sha256=digest, available_image_groups=len(groups),
        selected_image_ids=selected, excluded_image_ids=excluded,
        excluded_acquisition_sha256=sha256(PRIOR / 'acquisition.json'),
        class_csv_sha256=hashlib.sha256(classes).hexdigest(), boxes={i: groups[i] for i in selected},
        selection='Next sorted unseen TRAIN image IDs from pinned bounded window; not random or independent evaluation'))
    del raw
    print(f'Selected {len(selected)} unseen TRAIN IDs; streaming attribution metadata', flush=True)
    metadata, metadata_receipt = metadata_for(selected)
    write_json(output / 'attribution.json', metadata)
    write_json(output / 'metadata-source.json', metadata_receipt)
    (output / 'images').mkdir()
    images, failed, duplicates = [], [], []
    # map yields deterministic selection order while overlapping bounded public reads.
    with ThreadPoolExecutor(max_workers=4) as pool:
        for number, item in enumerate(pool.map(download, selected), 1):
            identifier = item['image_id']
            if 'error' in item:
                failed.append(item)
                print(f'Image {number}/{len(selected)} unavailable', flush=True)
                continue
            distance = holdout_distance(item['rgb'], prints)
            if distance <= 8:
                duplicates.append(dict(image_id=identifier, prior_train_hamming=distance))
                print(f'Image {number}/{len(selected)} excluded near-duplicate', flush=True)
                continue
            path = output / 'images' / (identifier + '.jpg')
            path.write_bytes(item['raw'])
            images.append(dict(image_id=identifier, path=f'images/{identifier}.jpg', sha256=sha256(path),
                size=[item['rgb'].shape[1], item['rgb'].shape[0]], url=item['url'], headers=item['headers'],
                annotations=groups[identifier], attribution=metadata[identifier], prior_train_hamming=distance))
            print(f'Image {number}/{len(selected)} downloaded', flush=True)
    receipt = dict(source='Open Images TRAIN, CVDF resized originals', images=images, failed=failed,
        rejected_prior_duplicates=duplicates, excluded_prior_acquisition_sha256=sha256(PRIOR / 'acquisition.json'),
        selection_sha256=sha256(output / 'selection.json'), attribution_sha256=sha256(output / 'attribution.json'),
        script_sha256=sha256(__file__), training_approved=False, review_required=True,
        per_image_license_verified=False, duplicate_screened=False,
        source_reported_annotation_license='CC BY 4.0', source_reported_image_license='CC BY 2.0',
        license_caveat='Source-page verification and visual labels still required; attribution is not an ownership warranty.',
        scope='Disjoint IDs and prior-TRAIN perceptual screen only. Reserved-source and within-batch duplicate screening still required.',
        test_evaluated=False, deployment_approved=False)
    write_json(output / 'acquisition.json', receipt)
    print(dict(downloaded=len(images), failed=len(failed), excluded_duplicates=len(duplicates),
               receipt_sha256=sha256(output / 'acquisition.json')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--limit', type=int, default=96)
    args = parser.parse_args()
    acquire(args.output, args.limit)
