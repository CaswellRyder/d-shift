"""Acquire new public TRAIN object photos for review, without assigning negatives."""
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
from pathlib import Path

from dtr.data import read_json, sha256, write_json
from scripts.acquire_blue_objects import CLASSES, object_groups, choose, blue_fraction
from scripts.acquire_openimages_balloons import BOX_URL, CLASS_URL, bounded_get, metadata_for
from scripts.acquire_more_openimages_balloons import download

OUTPUT = Path('data/raw/openimages-blue-objects-round3-20261011')


def previous_ids(acquisitions):
    """Exclude both retained and screened-out downloads, not just useful photos."""
    return {r['image_id'] for doc in acquisitions for key in ('images', 'screened_out')
            for r in doc.get(key, []) if 'image_id' in r}


def acquire():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    prior = sorted(Path('data/raw').glob('openimages*/acquisition.json'))
    excluded = previous_ids([read_json(p) for p in prior])
    classes, _ = bounded_get(CLASS_URL, 1 << 20)
    catalog = dict(csv.reader(io.StringIO(classes.decode())))
    if any(catalog.get(k) != v for k, v in CLASSES.items()):
        raise ValueError('Source category catalog changed')
    size = 64 << 20
    raw, headers = bounded_get(BOX_URL, size, {'Range': f'bytes=0-{size-1}'}, 206)
    if len(raw) != size or not headers.get('Content-Range', '').startswith(f'bytes 0-{size-1}/'):
        raise ValueError('Annotation range not honored')
    groups = object_groups(raw)
    selected = choose(groups, excluded, per_class=32)
    if not selected or len(selected) > 160 or set(selected) & excluded:
        raise ValueError('Invalid fresh bounded selection')
    OUTPUT.mkdir(parents=True)
    write_json(OUTPUT / 'selection.json', dict(annotations_url=BOX_URL, scan_bytes=size,
        scan_sha256=hashlib.sha256(raw).hexdigest(), headers=headers,
        class_csv_sha256=hashlib.sha256(classes).hexdigest(), classes=CLASSES,
        selected=selected, boxes={i: groups[i] for i in selected},
        excluded_acquisitions={str(p): sha256(p) for p in prior}, excluded_ids=sorted(excluded),
        selection='Same deterministic32-per-class quota, excluding every prior acquired or screened-out image. Known annotated balloons and incomplete range tail excluded. TRAIN only.'))
    del raw
    (OUTPUT / 'images').mkdir()
    kept, screened = [], []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for n, item in enumerate(pool.map(download, selected), 1):
            identifier = item['image_id']
            if 'error' in item:
                screened.append(item)
                continue
            stats = blue_fraction(item['rgb'], groups[identifier])
            if not any(s['blue_pixels'] >= 96 and s['fraction'] >= .08 for s in stats):
                screened.append(dict(image_id=identifier, reason='Insufficient blue inside upstream object boxes', stats=stats))
            else:
                path = OUTPUT / 'images' / (identifier + '.jpg')
                path.write_bytes(item['raw'])
                kept.append(dict(image_id=identifier, path='images/' + path.name, sha256=sha256(path),
                    url=item['url'], headers=item['headers'], size=[item['rgb'].shape[1], item['rgb'].shape[0]],
                    category=CLASSES[selected[identifier]], annotations=groups[identifier], blue_screen=stats))
            print(f'Object round3 {n}/{len(selected)}; retained {len(kept)}', flush=True)
    if not kept:
        raise ValueError('No blue-object review candidates')
    metadata, receipt = metadata_for([r['image_id'] for r in kept])
    write_json(OUTPUT / 'attribution.json', metadata)
    write_json(OUTPUT / 'metadata-source.json', receipt)
    for row in kept:
        row['attribution'] = metadata[row['image_id']]
    write_json(OUTPUT / 'acquisition.json', dict(images=kept, screened_out=screened,
        selection_sha256=sha256(OUTPUT / 'selection.json'), attribution_sha256=sha256(OUTPUT / 'attribution.json'),
        script_sha256=sha256(__file__), helpers_sha256={p: sha256(p) for p in (
            'scripts/acquire_blue_objects.py', 'scripts/acquire_more_openimages_balloons.py',
            'scripts/acquire_openimages_balloons.py')},
        source='Open Images TRAIN / CVDF resized originals', training_approved=False,
        review_required=True, per_image_license_verified=False, duplicate_screened=False,
        test_evaluated=False, deployment_approved=False,
        scope='Source class and blue color are candidate selection only, not background labels. Per-photo license verification, duplicate screening, full-frame and crop review required.'))
    print(dict(acquired=len(kept), screened_out=len(screened), receipt_sha256=sha256(OUTPUT / 'acquisition.json')))


if __name__ == '__main__':
    acquire()

