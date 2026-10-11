"""Source-capped TRAIN color-equivariance augmentation, never new real evidence."""
from collections import Counter, defaultdict

import numpy as np

LABELS = ('background', 'red_balloon', 'blue_balloon')


def swap(pixels, labels):
    pixels, labels = np.asarray(pixels), np.asarray(labels)
    if (pixels.ndim != 4 or pixels.shape[-1] != 3 or labels.shape != (len(pixels),)
            or pixels.dtype not in (np.dtype('uint8'), np.dtype('float32'))
            or not np.isfinite(pixels).all() or np.any((pixels < 0) | (pixels > 255))
            or not np.isin(labels, [0, 1, 2]).all()):
        raise ValueError('Require RGB0..255 and explicit three-class labels')
    return pixels[..., [2, 1, 0]].copy(), np.array([0, 2, 1], int)[labels.astype(int)]


def select_sources(rows):
    grouped, seen = defaultdict(list), {}
    for i, row in enumerate(rows):
        if row['split'] != 'train' or 'IMG_' in row['source_image'] or row['label'] not in LABELS:
            raise ValueError('TRAIN-only known-label source boundary')
        if row['view'] != 'original':
            continue
        key = row['sha256']
        if key in seen:
            if rows[seen[key]]['label'] != row['label']:
                raise ValueError('Conflicting exact-pixel labels')
            continue
        seen[key] = i
        grouped[row['source_image']].append(i)
    chosen = []
    for source in sorted(grouped):
        indexes = sorted(grouped[source], key=lambda i: rows[i]['sha256'])
        # Cover distinct classes before adding a second same-class crop.
        first = []
        for label in LABELS:
            matching = [i for i in indexes if rows[i]['label'] == label]
            if matching:
                first.append(matching[0])
        priority = sorted(first, key=lambda i: rows[i]['sha256'])
        priority += [i for i in indexes if i not in first]
        chosen.extend(priority[:2])
    groups = {label:sorted((i for i in chosen if rows[i]['label'] == label), key=lambda i: rows[i]['sha256'])
              for label in LABELS}
    count = min(len(groups['red_balloon']), len(groups['blue_balloon']))
    if count == 0:
        raise ValueError('Need both colors for balanced augmentation')
    selected = groups['background']+groups['red_balloon'][:count]+groups['blue_balloon'][:count]
    sources = Counter(rows[i]['source_image'] for i in selected)
    if max(sources.values()) > 2 or len({rows[i]['sha256'] for i in selected}) != len(selected):
        raise ValueError('Source cap or deduplication failed')
    return selected
