"""Acquire unseen public TRAIN balloon photos; blue pixels select review, not labels."""
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
from pathlib import Path

from dtr.data import read_json, sha256, write_json
from scripts.acquire_openimages_balloons import BALLOON, BOX_URL, CLASS_URL, bounded_get, complete_boxes, metadata_for
from scripts.acquire_more_openimages_balloons import download
from scripts.acquire_blue_objects import blue_fraction

OUTPUT = Path('data/raw/openimages-blue-balloon-candidates-20261011')


def unseen_selection(groups, excluded, limit=96):
    if type(limit) is not int or not 1 <= limit <= 96:
        raise ValueError('Require bounded1..96 selection')
    return sorted(set(groups) - set(excluded))[:limit]


def eligible_annotations(annotations):
    return [r for r in annotations if r['LabelName'] == BALLOON
            and r['IsDepiction'] == '0' and r['IsGroupOf'] == '0']


def acquire():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    prior = sorted(Path('data/raw').glob('openimages*/acquisition.json'))
    excluded, receipts = set(), {}
    for path in prior:
        doc, selection = read_json(path), read_json(path.parent / 'selection.json')
        if doc['selection_sha256'] != sha256(path.parent / 'selection.json'):
            raise ValueError('Prior acquisition changed')
        receipts[str(path)] = sha256(path)
        excluded.update(selection.get('selected_image_ids', selection.get('selected', {})))
        excluded.update(r['image_id'] for r in doc['images'])
    classes, _ = bounded_get(CLASS_URL, 1 << 20)
    if dict(csv.reader(io.StringIO(classes.decode()))).get(BALLOON) != 'Balloon':
        raise ValueError('Source class changed')
    size = 64 << 20
    raw, headers = bounded_get(BOX_URL, size, {'Range': f'bytes=0-{size-1}'}, 206)
    if len(raw) != size or not headers.get('Content-Range', '').startswith(f'bytes 0-{size-1}/'):
        raise ValueError('Incorrect annotation window')
    groups = complete_boxes(raw)
    selected = unseen_selection(groups, excluded)
    if not selected:
        raise ValueError('No unseen sources in bounded annotation window')
    OUTPUT.mkdir(parents=True)
    write_json(OUTPUT / 'selection.json', dict(annotations_url=BOX_URL, scan_bytes=size,
        scan_sha256=hashlib.sha256(raw).hexdigest(), headers=headers,
        class_csv_sha256=hashlib.sha256(classes).hexdigest(), available_image_groups=len(groups),
        selected_image_ids=selected, excluded_acquisitions=receipts,
        boxes={i: groups[i] for i in selected},
        selection='Sorted unseen TRAIN IDs, maximum96, excluding all prior attempted source IDs. Not independent evaluation.'))
    del raw
    (OUTPUT / 'images').mkdir()
    kept, rejected = [], []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for n, item in enumerate(pool.map(download, selected), 1):
            identifier = item['image_id']
            if 'error' in item:
                rejected.append(item)
                continue
            stats = blue_fraction(item['rgb'], eligible_annotations(groups[identifier]))
            if not any(s['blue_pixels'] >= 40 and s['fraction'] >= .03 for s in stats):
                rejected.append(dict(image_id=identifier, reason='No sufficient blue inside eligible source balloon boxes', stats=stats))
            else:
                path = OUTPUT / 'images' / (identifier + '.jpg')
                path.write_bytes(item['raw'])
                kept.append(dict(image_id=identifier, path=f'images/{identifier}.jpg', sha256=sha256(path),
                    size=[item['rgb'].shape[1], item['rgb'].shape[0]], url=item['url'], headers=item['headers'],
                    annotations=groups[identifier], blue_screen=stats))
            print(f'Candidate {n}/{len(selected)}; retained {len(kept)} for visual review', flush=True)
    if not kept:
        raise ValueError('No blue candidate photos; preserve screening evidence')
    metadata, receipt = metadata_for([r['image_id'] for r in kept])
    write_json(OUTPUT / 'attribution.json', metadata)
    write_json(OUTPUT / 'metadata-source.json', receipt)
    for row in kept:
        row['attribution'] = metadata[row['image_id']]
    write_json(OUTPUT / 'acquisition.json', dict(images=kept, screened_out=rejected,
        selection_sha256=sha256(OUTPUT / 'selection.json'), attribution_sha256=sha256(OUTPUT / 'attribution.json'),
        script_sha256=sha256(__file__), helpers_sha256={p: sha256(p) for p in
            ('scripts/acquire_openimages_balloons.py', 'scripts/acquire_more_openimages_balloons.py', 'scripts/acquire_blue_objects.py')},
        source='Open Images TRAIN / CVDF resized originals', training_approved=False,
        review_required=True, per_image_license_verified=False, duplicate_screened=False,
        test_evaluated=False, deployment_approved=False,
        scope='Upstream balloon annotations include unsuitable balloon types; blue may be background sky. Visual verification mandatory.'))
    print(dict(acquired=len(kept), screened_out=len(rejected)))


if __name__ == '__main__':
    acquire()
