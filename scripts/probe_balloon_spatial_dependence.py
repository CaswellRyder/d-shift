"""TRAIN-only spatial perturbation probe; transformed images are not labeled data.

Tile/pixel permutation preserves joint RGB histograms but also changes texture
and creates out-of-distribution inputs. It measures sensitivity, not a causal
proof of color-only reasoning or real-scene accuracy. No training admission.
"""
import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from dtr.data import sha256, write_json
from dtr.runtime import Predictor
from scripts.blue_object_round2_data import checked_extra as clutter_rows
from scripts.compare_current_blue_search import MODEL, MODEL_SHA
from scripts.indoor5_runtime_data import checked_extra as runtime_rows

OUTPUT = Path('research/balloon-red-blue-20261008/spatial-dependence-training-20261011.json')
VIEWS = ('original', 'horizontal_flip', 'rotate90', 'tiles8_seed47', 'tiles8_seed48',
         'pixels_seed47', 'pixels_seed48')


def transform(image, view):
    if image.shape != (64, 64, 3) or image.dtype != np.uint8 or view not in VIEWS:
        raise ValueError('Require64x64 RGB uint8 and a declared perturbation')
    if view == 'original':
        return image.copy()
    if view == 'horizontal_flip':
        return image[:, ::-1].copy()
    if view == 'rotate90':
        return np.rot90(image).copy()
    seed = int(view.rsplit('seed', 1)[1])
    rng = np.random.default_rng(seed)
    if view.startswith('pixels_'):
        return image.reshape(-1, 3)[rng.permutation(4096)].reshape(64, 64, 3).copy()
    tiles = image.reshape(8, 8, 8, 8, 3).transpose(0, 2, 1, 3, 4).reshape(64, 8, 8, 3)
    return tiles[rng.permutation(64)].reshape(8, 8, 8, 8, 3).transpose(0, 2, 1, 3, 4).reshape(64, 64, 3).copy()


def rgb_histogram(image):
    values = image.astype(np.uint32)
    keys = values[..., 0]*65536 + values[..., 1]*256 + values[..., 2]
    return np.sort(keys.ravel())


def probe():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    rows = [dict(r, source_set='runtime_indoor') for r in runtime_rows()]
    rows += [dict(r, source_set='blue_objects') for r in clutter_rows()]
    if Counter(r['label'] for r in rows) != dict(background=14, red_balloon=33, blue_balloon=29):
        raise ValueError('Reviewed source partition changed')
    if sha256(MODEL) != MODEL_SHA:
        raise ValueError('Frozen model changed')
    predictor = Predictor(MODEL, allow_unvalidated=True)
    if predictor.metadata['threshold'] != .8:
        raise ValueError('Operating point changed')
    probes = []
    for row in rows:
        if row['split'] != 'train' or 'IMG_' in row['source_image'] or sha256(row['file']) != row['sha256']:
            raise ValueError('Unsafe or changed training pixels')
        with Image.open(row['file']) as image:
            original = np.asarray(image.convert('RGB'))
            pixels = np.asarray(image.convert('RGB').resize((64, 64), Image.Resampling.BILINEAR))
        original_prediction = predictor.predict(original)
        histogram = rgb_histogram(pixels)
        views = {}
        for view in VIEWS:
            altered = transform(pixels, view)
            if not np.array_equal(rgb_histogram(altered), histogram):
                raise ValueError('Joint RGB histogram changed')
            prediction = predictor.predict(altered)
            views[view] = dict(scores=prediction['scores'], label=prediction['label'],
                accepted=prediction['accepted'], score=prediction['score'],
                pixel_sha256=hashlib.sha256(altered.tobytes()).hexdigest())
        if (views['original']['label'] != original_prediction['label']
                or views['original']['accepted'] != original_prediction['accepted']
                or not np.allclose(views['original']['scores'], original_prediction['scores'], atol=1e-6, rtol=0)):
            raise ValueError('Canonical resizing did not reproduce original crop prediction')
        probes.append(dict(id=row['id'], source_set=row['source_set'], source=row['source_image'],
            source_crop_sha256=row['sha256'], original_reviewed_label=row['label'], views=views))
    summaries = {}
    for label in ('background', 'red_balloon', 'blue_balloon'):
        group = [r for r in probes if r['original_reviewed_label'] == label]
        summaries[label] = {}
        for view in VIEWS:
            values = [r['views'][view] for r in group]
            summaries[label][view] = dict(rows=len(group),
                predicted_labels=dict(Counter(v['label'] for v in values)),
                accepted_balloon_count=sum(v['accepted'] and v['label'] != 'background' for v in values),
                accepted_original_class_count=sum(v['accepted'] and v['label'] == label for v in values),
                mean_foreground_score=float(np.mean([v['scores'][1]+v['scores'][2] for v in values])),
                mean_original_class_score=float(np.mean([v['scores'][('background','red_balloon','blue_balloon').index(label)] for v in values])))
    write_json(OUTPUT, dict(probes=probes, summary=summaries, rows=76,
        sources=len({r['source'] for r in probes}), views=list(VIEWS), seeds=[47,48],
        classifier=dict(path=str(MODEL), sha256=MODEL_SHA, metadata_sha256=sha256(MODEL.with_suffix('.json'))),
        script_sha256=sha256(__file__), source_receipts={p: sha256(p) for p in
            ('configs/indoor5-runtime-review-20261011.json', 'configs/blue-object-round2-review-20261011.json',
             'scripts/indoor5_runtime_data.py', 'scripts/blue_object_round2_data.py',
             'research/balloon-red-blue-20261008/indoor5-runtime-admission-20261011.json',
             'research/balloon-red-blue-20261008/blue-objects-round2-admission-20261011.json')},
        scope='TRAIN-only paired sensitivity diagnostic. Source labels apply to originals; transformed views have no ground-truth labels and are not admitted for training. Shuffling also changes texture/distribution; do not infer color-only causation or real accuracy.',
        training_performed=False, transformed_training_approved=False, test_evaluated=False,
        deployment_approved=False, qualification_achieved=False, pi_tested=False))
    print(summaries)


if __name__ == '__main__':
    probe()
