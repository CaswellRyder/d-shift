"""Duplicate-only quarantine screening against every reserved IMG family member."""
from pathlib import Path

import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, write_json
from scripts.prepare_blue_source_review import references
from scripts.prepare_indoor_balloon_review import fingerprint, holdout_distance, family

QUEUE = Path('data/engdes2-indoor-expansion5-review-20261011')
OUTPUT = Path('research/balloon-red-blue-20261008/indoor-expansion5-screen-20261011.json')


def screen():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    queue = read_json(QUEUE / 'review.json')
    prints, receipts = references()
    for name in ('engdes2-indoor-expansion3-review-20261011', 'engdes2-indoor-expansion4-review-20261011'):
        root = Path('data') / name
        path = root / 'review.json'
        receipts[str(path)] = sha256(path)
        for row in read_json(path)['frames']:
            image = root / row['path']
            if sha256(image) != row['sha256']:
                raise ValueError('Prior pixels changed')
            with Image.open(image) as im:
                prints.append(fingerprint(np.asarray(im.convert('RGB'))))
    selected, rows = [], []
    for frame in queue['frames']:
        path = QUEUE / frame['path']
        if sha256(path) != frame['sha256'] or family(frame['source']) is None:
            raise ValueError('Invalid training-family frame')
        with Image.open(path) as im:
            rgb = np.asarray(im.convert('RGB'))
        distance = holdout_distance(rgb, prints)
        internal = holdout_distance(rgb, selected) if selected else None
        eligible = distance > 8 and (internal is None or internal > 8)
        if eligible:
            selected.append(fingerprint(rgb))
        rows.append(dict(id=frame['id'], source=frame['source'], sha256=frame['sha256'],
            reference_hamming=distance, earlier_selected_hamming=internal, eligible=eligible,
            crop_ids=frame['crop_ids']))
    write_json(OUTPUT, dict(queue_sha256=sha256(QUEUE / 'review.json'), reference_inputs=receipts,
        reference_fingerprints=len(prints), frames=rows, script_sha256=sha256(__file__),
        helpers_sha256={p: sha256(p) for p in ('scripts/prepare_blue_source_review.py',
            'scripts/prepare_indoor_balloon_review.py')},
        scope='TRAIN quarantine only. Reserved images used only for automated duplicate checks, never fitting or labeling.',
        caveat='Perceptual hashes are heuristic; nearby training frames are not independent recordings.',
        training_approved=False, test_evaluated=False, deployment_approved=False))
    print(dict(eligible_frames=sum(r['eligible'] for r in rows),
               excluded_crops=[i for r in rows if not r['eligible'] for i in r['crop_ids']]))


if __name__ == '__main__':
    screen()
