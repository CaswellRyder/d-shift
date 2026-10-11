"""Preview deterministic paired color augmentation from already reviewed TRAIN crops."""
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from dtr.data import sha256, write_json
from scripts.window1_trial_data import tensors
from scripts.balloon_rb_swap import select_sources, swap, LABELS
from scripts.prepare_openimages_balloon_review import sheets

OUTPUT = Path('data/balloon-rb-swap-preview-20261011')


def prepare():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    data = tensors('hard')
    indexes = select_sources(data['rows'])
    x, y = data['x'][indexes], data['y'][indexes]
    changed, labels = swap(x, y)
    if Counter(y.tolist()) != Counter(labels.tolist()):
        raise ValueError('Paired class totals changed')
    preview = []
    for label in range(3):
        ids = np.flatnonzero(y == label)
        preview.extend(ids[np.linspace(0, len(ids)-1, min(6, len(ids)), dtype=int)].tolist())
    OUTPUT.mkdir(parents=True)
    records = []
    for i in preview:
        original, altered = x[i].astype(np.uint8), changed[i].astype(np.uint8)
        path = f'{i:03d}.png'
        Image.fromarray(np.concatenate([original, altered], axis=1)).save(OUTPUT/path)
        row = data['rows'][indexes[i]]
        records.append(dict(id=i, path=path, sha256=sha256(OUTPUT/path), source=row['source_image'],
            label_before=LABELS[y[i]], label_after=LABELS[labels[i]], original_index=indexes[i]))
    sheets(OUTPUT, 'paired', records, 3, (256, 160))
    write_json(OUTPUT/'review.json', dict(samples=records, selected_indexes=indexes,
        selected_sources=[{k:data['rows'][i][k] for k in ('source_image','sha256','label','view','split')} for i in indexes],
        label_counts=dict(Counter(LABELS[i] for i in y)), source_photos=len({data['rows'][i]['source_image'] for i in indexes}),
        samples_total=len(indexes), sources_sha256=data['sources_sha256'],
        helpers_sha256={p:sha256(p) for p in (__file__, 'scripts/balloon_rb_swap.py')},
        sheets={p.name:sha256(p) for p in OUTPUT.glob('*.jpg')},
        scope='Sampled visual sanity check of exact red/blue channel permutation, left original/right transformed. Entire selected inventory derives from known TRAIN labels; transformed labels follow red-blue permutation, not model inference. Synthetic views are not new independent photos or evaluation evidence.',
        training_approved=False, review_required=True, test_evaluated=False, deployment_approved=False))
    print(dict(selected=len(indexes), counts=dict(Counter(y.tolist())),
               sources=len({data['rows'][i]['source_image'] for i in indexes}), preview=len(preview)), flush=True)


if __name__ == '__main__':
    prepare()
