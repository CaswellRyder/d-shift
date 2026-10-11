"""Prepare every proposed synthetic hue training view for direct visual review."""
from pathlib import Path

import numpy as np
from PIL import Image

from dtr.data import sha256, write_json
from scripts.window1_trial_data import tensors
from scripts.balloon_hue_augmentation import select, recolor
from scripts.prepare_openimages_balloon_review import sheets

OUTPUT = Path('data/balloon-hue-review-20261011')


def prepare():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    data = tensors('hard')
    indexes = select(data)
    OUTPUT.mkdir(parents=True)
    records = []
    for j, i in enumerate(indexes):
        pixels, label, mask = recolor(data['x'][i], int(data['y'][i]))
        path = f'{j:03d}.png'
        Image.fromarray(np.concatenate([data['x'][i], pixels], axis=1).astype(np.uint8)).save(OUTPUT/path)
        row = data['rows'][i]
        records.append(dict(id=j, original_index=i, path=path, sha256=sha256(OUTPUT/path),
            source=row['source_image'], source_crop_sha256=row['sha256'], label_before=int(data['y'][i]),
            label_after=label, changed_fraction=float(mask.mean())))
    sheets(OUTPUT, 'paired', records, 3, (256, 160))
    write_json(OUTPUT/'review.json', dict(samples=records, selected_indexes=indexes,
        sources_sha256=data['sources_sha256'],
        helpers_sha256={p:sha256(p) for p in (__file__, 'scripts/balloon_hue_augmentation.py', 'scripts/balloon_rb_swap.py')},
        sheets={p.name:sha256(p) for p in OUTPUT.glob('*.jpg')},
        scope='All36 synthetic paired views require direct review. Left original, right hue-adjusted. Existing TRAIN labels only; no new real sources or evaluation evidence.',
        training_approved=False, test_evaluated=False, deployment_approved=False))
    print(dict(crops=len(records), sources=len({r['source'] for r in records})), flush=True)


if __name__ == '__main__':
    prepare()
