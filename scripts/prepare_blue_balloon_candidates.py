"""Licensed, duplicate-screened TRAIN review queue; never infer labels from blue pixels."""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, write_json
from scripts.acquire_blue_balloon_candidates import OUTPUT as ROOT, eligible_annotations
from scripts.acquire_blue_objects import blue_fraction
from scripts.prepare_blue_source_review import references, crop_box
from scripts.prepare_openimages_balloon_review import extra_references, checked_path, sheets
from scripts.prepare_indoor_balloon_review import fingerprint, holdout_distance

OUTPUT = Path('data/blue-balloon-candidates-review-20261011')


def prepare():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    acquisition, licenses = read_json(ROOT / 'acquisition.json'), read_json(ROOT / 'license-verification.json')
    if (licenses['acquisition_sha256'] != sha256(ROOT / 'acquisition.json')
            or acquisition['script_sha256'] != sha256('scripts/acquire_blue_balloon_candidates.py')
            or any(sha256(p) != h for p, h in acquisition['helpers_sha256'].items())):
        raise ValueError('Acquisition identity changed')
    for key, name in (('selection_sha256', 'selection.json'), ('attribution_sha256', 'attribution.json')):
        if acquisition[key] != sha256(ROOT / name):
            raise ValueError('Acquisition metadata changed')
    verified = {r['image_id'] for r in licenses['results'] if r['verified'] is True}
    prints, receipts = references()
    extra_references(prints, receipts)
    for p, h in read_json(ROOT / 'selection.json')['excluded_acquisitions'].items():
        if sha256(p) != h:
            raise ValueError('Previous acquisition changed')
        receipts[p] = h
        for r in read_json(p)['images']:
            with Image.open(checked_path(Path(p).parent, r['path'], r['sha256'])) as image:
                prints.append(fingerprint(np.asarray(image.convert('RGB'))))
    frames, samples, rejected, selected = [], [], [], []
    OUTPUT.mkdir(parents=True)
    (OUTPUT / 'frames').mkdir()
    (OUTPUT / 'crops').mkdir()
    for source in acquisition['images']:
        if source['image_id'] not in verified:
            rejected.append(dict(image_id=source['image_id'], reason='License not verified'))
            continue
        if source['attribution']['Subset'] != 'train':
            raise ValueError('Non-TRAIN source')
        with Image.open(checked_path(ROOT, source['path'], source['sha256'])) as image:
            rgb = np.asarray(image.convert('RGB'))
        distance = holdout_distance(rgb, prints)
        internal = holdout_distance(rgb, selected) if selected else None
        if distance <= 8 or (internal is not None and internal <= 8):
            rejected.append(dict(image_id=source['image_id'], reason='Perceptual duplicate',
                reference_hamming=distance, selected_hamming=internal))
            continue
        scan = cv2.resize(rgb, (320, 240), interpolation=cv2.INTER_AREA)
        annotations = eligible_annotations(source['annotations'])
        stats = blue_fraction(rgb, annotations)
        boxes = [[float(r[k]) * scale for k, scale in zip(('XMin', 'YMin', 'XMax', 'YMax'), (320, 240, 320, 240))]
                 for r, stat in zip(annotations, stats, strict=True)
                 if stat['blue_pixels'] >= 40 and stat['fraction'] >= .03][:24]
        i = len(frames)
        path = f'frames/{i:03d}.png'
        Image.fromarray(scan).save(OUTPUT / path)
        frames.append(dict(id=i, path=path, sha256=sha256(OUTPUT / path), image_id=source['image_id'],
            original_path=source['path'], original_sha256=source['sha256'], boxes=boxes,
            attribution=source['attribution'], reference_hamming=distance, selected_hamming=internal))
        selected.append(fingerprint(rgb))
        for box in boxes:
            bounds = crop_box(box)
            a, b, c, d = bounds
            if min(c - a, d - b) < 4:
                continue
            path = f'crops/{len(samples):03d}.png'
            Image.fromarray(scan[b:d, a:c]).save(OUTPUT / path)
            samples.append(dict(id=len(samples), frame_id=i, path=path, sha256=sha256(OUTPUT / path),
                box=box, crop_box=bounds, source_image='openimages/train/' + source['image_id'],
                split='train', label='unreviewed'))
    sheets(OUTPUT, 'frames', frames, 3, (320, 270))
    sheets(OUTPUT, 'crops', samples, 6, (160, 160))
    write_json(OUTPUT / 'review.json', dict(frames=frames, samples=samples, rejected=rejected,
        acquisition_sha256=sha256(ROOT / 'acquisition.json'), license_sha256=sha256(ROOT / 'license-verification.json'),
        script_sha256=sha256(__file__), reference_inputs=receipts,
        helpers_sha256={p: sha256(p) for p in ('scripts/acquire_blue_balloon_candidates.py', 'scripts/acquire_blue_objects.py',
            'scripts/prepare_blue_source_review.py', 'scripts/prepare_openimages_balloon_review.py', 'scripts/prepare_indoor_balloon_review.py')},
        sheets={p.name: sha256(p) for p in OUTPUT.glob('*.jpg')},
        training_approved=False, review_required=True, heldout_pixels_used_for_duplicate_screening_only=True,
        test_evaluated=False, deployment_approved=False,
        scope='Source boxes and blue pixels select candidates only; may include sky, hot-air balloons or non-target shapes. Manual visual labels required.'))
    print(dict(frames=len(frames), crops=len(samples), rejected=len(rejected), queue_sha256=sha256(OUTPUT / 'review.json')))


if __name__ == '__main__':
    prepare()
