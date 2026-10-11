"""Expand real TRAIN source diversity with bounded acquisition and explicit quarantine.

No prior source ID is retried, including previously screened-out images. Blue
pixels only select candidates: source licenses, duplicates and labels still need
verification. No validation/test image download or runtime model change.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
from pathlib import Path

from dtr.data import sha256, write_json
from scripts.acquire_blue_balloon_candidates import eligible_annotations, unseen_selection
from scripts.acquire_blue_objects import blue_fraction
from scripts.acquire_more_openimages_balloons import download
from scripts.acquire_openimages_balloons import (
    BALLOON, BOX_URL, CLASS_URL, bounded_get, metadata_for,
)


from scripts.acquire_blue_balloon_batch import prior_sources
from scripts.openimages_annotation_window import complete_window


def acquire(output, scan_bytes=128 << 20, limit=96, start_byte=256 << 20):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if (type(scan_bytes) is not int or not 64 << 20 <= scan_bytes <= 256 << 20
            or type(start_byte) is not int or not 256 << 20 <= start_byte <= 4 << 30):
        raise ValueError('Require bounded64..256MiB interior window starting256MiB..4GiB')
    unseen_selection({}, set(), limit)  # Validate count before any network request.
    excluded, receipts = prior_sources('data/raw')
    classes, _ = bounded_get(CLASS_URL, 1 << 20)
    if dict(csv.reader(io.StringIO(classes.decode()))).get(BALLOON) != 'Balloon':
        raise ValueError('Source class changed')
    header_raw, header_response = bounded_get(BOX_URL, 4096, {'Range':'bytes=0-4095'}, 206)
    if len(header_raw) != 4096 or not header_response.get('Content-Range', '').startswith('bytes 0-4095/'):
        raise ValueError('Incorrect annotation header response')
    header = header_raw.split(b'\n', 1)[0]
    end_byte = start_byte+scan_bytes-1
    raw, headers = bounded_get(BOX_URL, scan_bytes, {'Range': f'bytes={start_byte}-{end_byte}'}, 206)
    if (len(raw) != scan_bytes
            or not headers.get('Content-Range', '').startswith(f'bytes {start_byte}-{end_byte}/')):
        raise ValueError('Incorrect annotation window')
    if headers.get('ETag') != header_response.get('ETag') or not headers.get('ETag'):
        raise ValueError('Annotation object changed between header and window')
    groups = complete_window(raw, header)
    selected = unseen_selection(groups, excluded, limit)
    if not selected:
        raise ValueError('No unseen TRAIN source in this bounded window')
    output.mkdir(parents=True)
    write_json(output / 'selection.json', dict(
        annotations_url=BOX_URL, scan_bytes=scan_bytes, start_byte=start_byte,
        end_byte=end_byte, header_sha256=hashlib.sha256(header).hexdigest(),
        header_response=header_response, edge_policy='Discard partial edge lines and both edge image groups',
        scan_sha256=hashlib.sha256(raw).hexdigest(), headers=headers,
        class_csv_sha256=hashlib.sha256(classes).hexdigest(), available_image_groups=len(groups),
        selected_image_ids=selected, excluded_acquisitions=receipts,
        boxes={i: groups[i] for i in selected},
        selection='Sorted unseen TRAIN IDs; excludes all earlier attempted sources. Not independent evaluation.'))
    del raw
    print(dict(groups=len(groups), selected=len(selected), excluded=len(excluded)), flush=True)
    (output / 'images').mkdir()
    kept, rejected = [], []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for n, item in enumerate(pool.map(download, selected), 1):
            identifier = item['image_id']
            if 'error' in item:
                rejected.append(item)
                continue
            stats = blue_fraction(item['rgb'], eligible_annotations(groups[identifier]))
            if not any(s['blue_pixels'] >= 40 and s['fraction'] >= .03 for s in stats):
                rejected.append(dict(image_id=identifier, reason='Insufficient blue in eligible source boxes', stats=stats))
            else:
                path = output / 'images' / (identifier + '.jpg')
                path.write_bytes(item['raw'])
                kept.append(dict(image_id=identifier, path=f'images/{identifier}.jpg', sha256=sha256(path),
                    size=[item['rgb'].shape[1], item['rgb'].shape[0]], url=item['url'], headers=item['headers'],
                    annotations=groups[identifier], blue_screen=stats))
            print(f'Candidate {n}/{len(selected)}; retained {len(kept)} for review', flush=True)
    metadata, receipt = metadata_for([r['image_id'] for r in kept]) if kept else ({}, {'empty': True})
    write_json(output / 'attribution.json', metadata)
    write_json(output / 'metadata-source.json', receipt)
    for row in kept:
        row['attribution'] = metadata[row['image_id']]
    write_json(output / 'acquisition.json', dict(images=kept, screened_out=rejected,
        selection_sha256=sha256(output / 'selection.json'), attribution_sha256=sha256(output / 'attribution.json'),
        script_path=__file__, script_sha256=sha256(__file__), helpers_sha256={p: sha256(p) for p in (
            'scripts/acquire_blue_balloon_candidates.py', 'scripts/acquire_openimages_balloons.py',
            'scripts/openimages_annotation_window.py', 'scripts/acquire_blue_balloon_batch.py',
            'scripts/acquire_more_openimages_balloons.py', 'scripts/acquire_blue_objects.py')},
        source='Open Images TRAIN / CVDF resized originals', training_approved=False,
        review_required=True, per_image_license_verified=False, duplicate_screened=False,
        test_evaluated=False, deployment_approved=False,
        scope='Candidate acquisition only. Blue may be sky or non-target balloon types; visual verification mandatory.'))
    print(dict(acquired=len(kept), screened_out=len(rejected)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--scan-bytes', type=int, default=128 << 20)
    parser.add_argument('--limit', type=int, default=96)
    parser.add_argument('--start-byte', type=int, default=256 << 20)
    args = parser.parse_args()
    acquire(args.output, args.scan_bytes, args.limit, args.start_byte)

